# A6000 / 100 GB Local Custom Corpus Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable local pipeline that processes organizer ZIP files
in offset/limit archive windows, runs every group of at most eight videos from
native keyframe extraction through specialist enrichment, embeddings, and three
loadable retrieval indexes, and stays within a measured A6000 / 100 GB / 6-vCPU
resource envelope.

**Architecture:** The runbook remains the sole ordered ZIP source plan. Python
downloads and extracts one archive, plans canonical groups of at most eight,
and drives stage-scoped local model processes. Each group is atomically
committed under `artifacts/<version>/batches/` only after Caption, OCR, Objects,
FrameContext, visual/context embeddings, reused ASR vectors, and visual/context/
ASR indexes validate. ZIPs, source MP4s, OCR scratch, and native staging are
then cleaned by exact path; durable keyframes and compact artifacts/indexes are
retained locally. No task performs S3 or other cloud publication.

**Tech Stack:** Python 3.11, Pydantic, pandas/pyarrow, NumPy, pytest, Qwen3-VL,
Florence2, YOLOE, SigLIP2, BGE-M3, FAISS, C++17, FFmpeg, Bash, NVIDIA A6000,
and standard Linux disk/process tools.

**Spec:**
`docs/superpowers/specs/2026-08-28-a6000-100gb-custom-pipeline-design.md`

## Global Constraints

- Work only in this repository. Never delete `data/`, `artifacts/`, or
  `artifacts_legacy/`.
- Inspect `git status` before every task and preserve unrelated user/teammate
  changes, especially object-detection and runbook work.
- Keep the BTC-keyframe profile unchanged. All new behavior belongs to the
  explicit custom raw-video profile.
- Organizer ZIP URLs in the runbook are the only video source. Do not invoke
  yt-dlp or acquire from media-info `watch_url`.
- Preserve `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms` through every
  table, mapping, manifest, and index.
- Keep the native competition formula exactly
  `floor(ceil(avg_fps_of_video) * timestamp_ms / 1000)`; Python only rejects
  mismatches.
- `offset` and `limit` select zero-based archive positions and archive count.
  They never select videos.
- Process every validated member of each selected archive in canonical groups
  of at most eight.
- Run one GPU-heavy model process at a time on the A6000.
- Initial CPU limits are two FFmpeg processes × two threads, three GPU-stage
  image workers, and six PyArrow/FAISS threads only in CPU-only stages.
- Enforce a 15 GiB free reserve and 30 GiB active-tree cap using measured bytes.
- Retain durable keyframes and final artifacts. Delete only ZIPs, source MP4s,
  OCR scratch, native staging, and exact batch-temporary paths after a validated
  local batch commit.
- Reuse validated ASR transcripts/vectors. Do not run ASR inference or
  re-embedding in this pipeline.
- Keep Caption, raw/normalized OCR, Objects, FrameContext, and provenance as
  independent artifacts. FrameContext excludes ASR.
- Write state, manifests, reports, Parquet, NumPy, and indexes atomically.
- Cloud upload/sync is operator-owned and outside this plan. No implementation
  task may require cloud credentials or remote state.
- Add useful module/class/public-function docstrings. Comments explain identity,
  resource, resume, and cleanup invariants.
- Run focused tests after every task. Do not start the real A6000 pilot until
  local release gates pass.

## Target File Map

```text
src/hcmai/data/custom_pipeline/
├── __init__.py       # stable local-pipeline contracts
├── config.py         # A6000, CPU, disk, and archive-window configuration
├── contracts.py      # run, stage, batch, and local artifact contracts
├── state.py          # atomic archive/batch/video resume state
├── disk.py           # real-byte measurement and admission
├── archive.py        # one-ZIP download, safe extraction, inventory, grouping
├── asr.py            # reusable transcript/vector lineage validation
├── shards.py         # per-video shards and three per-batch indexes
├── stages.py         # isolated local stages, CPU limits, OOM backoff
├── commit.py         # atomic local batch commit and exact ephemeral cleanup
├── finalize.py       # streaming corpus compaction and global indexes
└── runner.py         # preflight, process-archive, status, finalize composition

scripts/prepare_custom_pipeline.py
docs/runbooks/bootstrap_and_run_custom_pipeline.sh

tests/data/custom_pipeline/
├── test_config.py
├── test_state.py
├── test_disk.py
├── test_archive.py
├── test_asr.py
├── test_shards.py
├── test_stages.py
├── test_commit.py
├── test_finalize.py
└── test_runner.py
```

---

## Task 1: Define the local A6000 pipeline configuration and contracts

**Files:**

- Create: `src/hcmai/data/custom_pipeline/__init__.py`
- Create: `src/hcmai/data/custom_pipeline/contracts.py`
- Create: `src/hcmai/data/custom_pipeline/config.py`
- Modify: `configs/prepare.yaml`
- Create: `tests/data/custom_pipeline/test_config.py`

**Interfaces:**

- `DiskBudgetConfig(min_free_gib=15, max_active_gib=30, max_archive_download_gib=20, max_archive_uncompressed_gib=25)` exposes exact
  byte properties.
- `SchedulingConfig(max_videos_per_batch=8, extractor_processes=2, ffmpeg_threads_per_process=2, image_workers=3, prefetch_batches=2, cpu_only_threads=6)` rejects CPU oversubscription against `available_cpus=6`.
- `StageBatchConfig(caption=8, ocr=32, objects=32, visual=128, context=128, minimum=1)` contains pilot starting values.
- `ArchivePlanEntry(position, archive_id, url)` and `ArchivePlan(entries, digest)` preserve the complete ordered HTTPS ZIP plan.
- `ArchiveWorkWindow(offset=0, limit=None)` selects a non-empty zero-based
  archive slice; limit is a positive archive count.
- `RunIdentity` freezes version, source, frame-store ID, media-info digest,
  complete archive-plan digest, artifact-config fingerprint, model revisions,
  and ASR lineage. Work windows and effective resource values are operational
  attempt history, not artifact identity.

- [ ] **Step 1: Write failing configuration tests**

  Cover GiB conversion, invalid reserves/caps, six-CPU oversubscription,
  non-positive worker/model batches, duplicate/non-HTTPS/non-ZIP URLs, ordered
  plan digest, default window, offset 2 + limit 3 → positions 2–4, omitted limit,
  out-of-range offset, and proof that no cloud destination exists in serialized
  config or run identity.
- [ ] **Step 2: Run the focused tests and confirm missing imports/contracts**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/data/custom_pipeline/test_config.py -q
  ```
- [ ] **Step 3: Implement the smallest immutable contracts**

  Reuse repository config and atomic-I/O helpers. Keep resource values explicit
  and add module/public API docstrings.
- [ ] **Step 4: Add the custom A6000 YAML profile**

  Add disk, scheduling, model batches, local roots, and A6000 device/dtype
  settings. Label every performance value pilot-tunable and do not alter BTC
  defaults.
- [ ] **Step 5: Run focused and existing config tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_config.py tests/test_prepare_config.py -q
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add configs/prepare.yaml src/hcmai/data/custom_pipeline \
    tests/data/custom_pipeline/test_config.py
  git commit -m "feat(data): define local A6000 pipeline contracts"
  ```

---

## Task 2: Persist local archive, batch, and video resume state

**Files:**

- Create: `src/hcmai/data/custom_pipeline/state.py`
- Create: `tests/data/custom_pipeline/test_state.py`
- Modify: `src/hcmai/data/custom_pipeline/__init__.py`

**Interfaces:**

- `ArchiveStage`: `pending`, `downloading`, `downloaded`, `extracted`,
  `processing`, `complete`, `cleaned`.
- `BatchStage`: `planned`, `extracted`, `artifacts_complete`,
  `indexes_complete`, `committed`, `ephemeral_cleaned`.
- `VideoStage`: `pending`, `source_ready`, `extracted`, `captioned`,
  `ocr_complete`, `objects_complete`, `context_complete`,
  `embeddings_complete`, `local_complete`.
- `PipelineStateStore.create_or_resume_run(identity, work_window)` rejects a
  changed run identity and appends accepted work-window attempts.
- One atomic JSON record is stored per archive, deterministic batch, and video.
- `require_ephemeral_cleanup_allowed(batch_id)` accepts only `committed` or
  `ephemeral_cleaned` and verifies all contained videos are `local_complete`.

- [ ] **Step 1: Write failing state-machine tests**

  Test atomic create/resume, changed identity rejection, adjacent work windows,
  gap rejection, cleaned overlap replay, ordered transitions, skipped/reversed
  transition rejection, deterministic batch IDs, eight-video ceiling, bounded
  failure history, and cleanup forbidden before local commit.
- [ ] **Step 2: Run the tests and confirm failure**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/data/custom_pipeline/test_state.py -q
  ```
- [ ] **Step 3: Implement atomic state without modifying native JSON**

  Store files under `runs/<version>/state/{archives,batches,videos}/`. Use safe
  IDs, expected-state compare/advance, and idempotent identical replay.
- [ ] **Step 4: Run state and native-wrapper regressions**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_state.py tests/data/test_custom_state.py -q
  ```
- [ ] **Step 5: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline tests/data/custom_pipeline/test_state.py
  git commit -m "feat(data): persist local batch pipeline state"
  ```

---

## Task 3: Implement one-archive acquisition and disk-bounded native extraction

**Files:**

- Create: `src/hcmai/data/custom_pipeline/disk.py`
- Create: `src/hcmai/data/custom_pipeline/archive.py`
- Create: `tests/data/custom_pipeline/test_disk.py`
- Create: `tests/data/custom_pipeline/test_archive.py`
- Create: `src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/disk_guard.hpp`
- Create: `src/hcmai/data/cpp/keyframes_extraction/src/disk_guard.cpp`
- Create: `src/hcmai/data/cpp/keyframes_extraction/tests/test_disk_guard.cpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/types.hpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/extractor.hpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/state.hpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/src/config.cpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/src/extractor.cpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/src/jsonl.cpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/src/main.cpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/src/state.cpp`
- Modify: `src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt`
- Modify: `scripts/extract_custom_keyframes.py`
- Modify: `src/hcmai/data/ingestion/custom_manifest.py`
- Modify: `tests/data/test_custom_manifest.py`
- Modify: `tests/scripts/test_custom_extraction_cli.py`

**Interfaces:**

- `measure_tree_bytes(path)` counts unique regular-file inodes and ignores
  symlinks.
- `snapshot_disk(run_root, active_root)` reports real free and active bytes.
- `require_write_capacity(...)` enforces both 15 GiB reserve and 30 GiB cap.
- `download_archive(...)` uses shell-free resumable curl into `.part`, bounded
  retries, byte ceilings, and incremental disk checks.
- `inspect_archive(...)` rejects traversal, absolute paths, links, duplicate
  members/IDs, non-MP4 payloads, and declared-size abuse.
- `extract_archive_atomically(...)` validates actual extracted sizes and deletes
  the ZIP only after `archive_manifest.json` commits.
- `plan_archive_batches(inventory, batch_size=8)` includes every member and
  returns canonical groups of eight plus final remainder.
- `stage_archive_source_links(...)` uses same-filesystem hard links only.
- Native `DiskBudgetGuard` reserves a conservative JPEG write upper bound.
- Native extraction requires local `--source-root`; network/watch-URL fields are
  removed from active config, state, and manifests.

- [ ] **Step 1: Write failing Python/native boundary tests**

  Cover one-byte disk boundaries, hard-link accounting, unsafe ZIPs, nested
  `Lxx/Lxx_Vnnn.mp4`, resumed curl argv, actual-size mismatch, ZIP deletion
  timing, groups `[8, 8, remainder]`, hard-link-only staging, two-extractor CPU
  limit, and native frame-write reserve failure.
- [ ] **Step 2: Run failing focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_disk.py \
    tests/data/custom_pipeline/test_archive.py \
    tests/data/test_custom_manifest.py \
    tests/scripts/test_custom_extraction_cli.py -q
  cmake --build build/keyframes_extraction --parallel
  ctest --test-dir build/keyframes_extraction -R disk_guard --output-on-failure
  ```
- [ ] **Step 3: Implement disk measurement and safe archive handling**

  Diagnostics include operation, reserve, active/cap, requested bytes, ZIP
  bytes, and declared extraction bytes. Never expose partial extraction as
  committed.
- [ ] **Step 4: Implement local-only native extraction and CPU limits**

  Keep exact sampling/JPEG/identity behavior. Pass two-process/two-thread limits
  from Python without adding a generic scheduler.
- [ ] **Step 5: Run Python and complete native suites**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_disk.py \
    tests/data/custom_pipeline/test_archive.py \
    tests/data/test_custom_manifest.py \
    tests/scripts/test_custom_extraction_cli.py -q
  cmake -S src/hcmai/data/cpp/keyframes_extraction \
    -B build/keyframes_extraction -DCMAKE_BUILD_TYPE=Release
  cmake --build build/keyframes_extraction --parallel
  ctest --test-dir build/keyframes_extraction --output-on-failure
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/disk.py \
    src/hcmai/data/custom_pipeline/archive.py \
    src/hcmai/data/cpp/keyframes_extraction \
    src/hcmai/data/ingestion/custom_manifest.py \
    scripts/extract_custom_keyframes.py \
    tests/data/custom_pipeline/test_disk.py \
    tests/data/custom_pipeline/test_archive.py \
    tests/data/test_custom_manifest.py \
    tests/scripts/test_custom_extraction_cli.py
  git commit -m "feat(data): extract archive batches within local disk limits"
  ```

---

## Task 4: Validate reusable ASR transcripts and vectors

**Files:**

- Create: `src/hcmai/data/custom_pipeline/asr.py`
- Create: `tests/data/custom_pipeline/test_asr.py`
- Modify: `src/hcmai/data/ingestion/custom_enrichment.py`
- Modify: `tests/data/test_custom_enrichment.py`

**Interfaces:**

- `ASRReuseBundle(transcripts_root, index_root, video_ids, transcript_fingerprint, index_fingerprint, segment_count)`.
- `validate_asr_source(...)` validates transcript manifests, segment identity,
  timestamps, vector mapping/count/dimension/finiteness, and immutable lineage.
- `require_asr_video_coverage(bundle, archive_video_ids)` runs after archive
  inventory and before extraction.
- No function in this package invokes transcription or a text encoder.

- [ ] **Step 1: Write failing ASR lineage/coverage tests**

  Cover valid reuse, missing archive video, missing manifest, duplicate segment,
  invalid interval, mapping/vector mismatch, corrupt checksum, accepted unrelated
  source videos, and deterministic fingerprints.
- [ ] **Step 2: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/data/custom_pipeline/test_asr.py -q
  ```
- [ ] **Step 3: Implement through existing transcript/index loaders**

  Preserve segment-native identity and make `source = reused_existing_asr`
  explicit in local manifests.
- [ ] **Step 4: Run ASR and handoff regressions**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_asr.py \
    tests/data/test_custom_enrichment.py \
    tests/retrieval/test_asr_segment_retriever.py -q
  ```
- [ ] **Step 5: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/asr.py \
    src/hcmai/data/ingestion/custom_enrichment.py \
    tests/data/custom_pipeline/test_asr.py tests/data/test_custom_enrichment.py
  git commit -m "feat(data): validate reusable local ASR artifacts"
  ```

---

## Task 5: Make Caption and OCR explicit local A6000 stages

**Files:**

- Modify: `src/hcmai/data/enrichment/caption/generator.py`
- Modify: `scripts/generate_ocr_enrichment.py`
- Modify: `tests/scripts/test_enrichment_clis.py`
- Modify: `tests/test_caption.py`
- Modify: `tests/test_ocr.py`
- Modify: `tests/data/enrichment/test_caption_evidence.py`
- Modify: `tests/data/enrichment/test_ocr_evidence.py`
- Modify: `tests/test_remote_preparation_adapters.py`

**Interfaces:**

- Caption and OCR CLIs retain compatibility but expose explicit
  `--execution-backend local`, `--batch-size`, and `--image-workers` overrides.
- The custom runner always selects local adapters, starts one model process,
  uses at most three image workers, and releases it at stage process exit.
- Existing object detection is consumed through its current local CLI/package;
  preserve teammate-owned detector semantics.

- [ ] **Step 1: Write failing parser and adapter-selection tests**

  Verify local/remote compatibility, custom-run local selection, batch and
  worker validation, unchanged identity arguments, and no hidden gateway start.
- [ ] **Step 2: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/scripts/test_enrichment_clis.py tests/test_caption.py tests/test_ocr.py \
    tests/data/enrichment/test_caption_evidence.py \
    tests/data/enrichment/test_ocr_evidence.py \
    tests/test_remote_preparation_adapters.py -q
  ```
- [ ] **Step 3: Implement explicit local adapter and worker overrides**

  Keep model imports inside selected branches. Do not change prompts, output
  schemas, image sizes, or model revisions in this task.
- [ ] **Step 4: Run tests and compile modified CLIs**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/scripts/test_enrichment_clis.py tests/test_caption.py tests/test_ocr.py \
    tests/data/enrichment/test_caption_evidence.py \
    tests/data/enrichment/test_ocr_evidence.py \
    tests/test_remote_preparation_adapters.py -q
  PYTHONPATH=.:src aic/bin/python -m compileall -q \
    src/hcmai/data/enrichment/caption/generator.py \
    scripts/generate_ocr_enrichment.py
  ```
- [ ] **Step 5: Commit**

  ```bash
  git add src/hcmai/data/enrichment/caption/generator.py \
    scripts/generate_ocr_enrichment.py tests/scripts/test_enrichment_clis.py \
    tests/test_caption.py tests/test_ocr.py \
    tests/data/enrichment/test_caption_evidence.py \
    tests/data/enrichment/test_ocr_evidence.py \
    tests/test_remote_preparation_adapters.py
  git commit -m "feat(enrichment): run isolated local A6000 stages"
  ```

---

## Task 6: Build per-video shards and three indexes for every batch

**Files:**

- Create: `src/hcmai/data/custom_pipeline/shards.py`
- Create: `tests/data/custom_pipeline/test_shards.py`
- Modify: `src/hcmai/retrieval/embedding/artifacts.py`
- Modify: `src/hcmai/retrieval/retriever/text/retriever.py`
- Modify: `src/hcmai/retrieval/retriever/dense/index.py`
- Modify: `src/hcmai/retrieval/retriever/segment/index.py`
- Modify: `tests/retrieval/test_visual_embedding_resume.py`
- Modify: `tests/retrieval/test_context_index.py`
- Modify: `tests/retrieval/test_segment_dense_index.py`
- Modify: `tests/retrieval/test_asr_segment_retriever.py`

**Interfaces:**

- `split_batch_artifacts_by_video(...)` writes canonical Caption, OCR frame/
  region, Object frame/detection, FrameContext, visual, and context shards.
- `validate_video_shard(...)` enforces exact frame coverage while allowing zero
  OCR regions or object detections.
- Per-video visual/context vector files have aligned canonical mappings but no
  redundant per-video FAISS index.
- `build_batch_index_bundle(batch_id, video_ids, video_shards, asr_bundle, output_root)` concatenates the ordered group, subsets persisted ASR vectors,
  builds visual/context `DenseIndex` and ASR `SegmentDenseIndex`, then
  checksum-loads all three.
- `BatchIndexInventory` records ordered IDs, lineage, counts, dimensions, paths,
  sizes, and SHA-256.

- [ ] **Step 1: Write failing shard and mapping tests**

  Cover exact ordered coverage, duplicate/missing/foreign IDs, empty child
  tables, finite dimensions, contiguous mapping positions, custom frames without
  `keyframe_order`, deterministic inventory, groups of eight, and remainder.
- [ ] **Step 2: Write failing precomputed batch-index tests**

  Use fake encoders and persisted ASR vectors. Assert all three indexes load,
  contain only batch video IDs, preserve canonical mappings, and never invoke
  ASR/text encoding for reused ASR.
- [ ] **Step 3: Run focused tests and confirm failures**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_shards.py \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_segment_dense_index.py \
    tests/retrieval/test_asr_segment_retriever.py -q
  ```
- [ ] **Step 4: Implement reusable embedding-shard functions**

  Reuse `EmbeddingArtifactBuilder`, existing text normalization, `DenseIndex`,
  and `SegmentDenseIndex`. Do not duplicate model adapters or index formats.
- [ ] **Step 5: Implement deterministic split, inventory, and index bundle**

  Order by `(video_id, timestamp_ms, frame_id)`, rewrite batch-local mapping
  indices, and publish the local index directory atomically only after all three
  loads pass.
- [ ] **Step 6: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_shards.py \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_segment_dense_index.py \
    tests/retrieval/test_asr_segment_retriever.py -q
  ```
- [ ] **Step 7: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/shards.py \
    src/hcmai/retrieval/embedding/artifacts.py \
    src/hcmai/retrieval/retriever/text/retriever.py \
    src/hcmai/retrieval/retriever/dense/index.py \
    src/hcmai/retrieval/retriever/segment/index.py \
    tests/data/custom_pipeline/test_shards.py \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_segment_dense_index.py \
    tests/retrieval/test_asr_segment_retriever.py
  git commit -m "feat(data): build three local indexes per video batch"
  ```

---

## Task 7: Run isolated stages with A6000 OOM backoff and six-CPU limits

**Files:**

- Create: `src/hcmai/data/custom_pipeline/stages.py`
- Create: `tests/data/custom_pipeline/test_stages.py`

**Interfaces:**

- `StageCommand(name, argv, initial_batch_size, output_path, image_workers=3, cpu_threads=None)` uses argv without shell interpolation.
- `run_stage(...)` recognizes CUDA OOM diagnostics, halves only model batch
  size, stops at one, and records attempts/timing/resource snapshots.
- `run_batch_stages(context)` executes extraction, Caption, OCR, OCR-scratch
  cleanup, Objects, FrameContext, visual embedding, context embedding, and three
  batch indexes in strict order.
- Extraction uses two processes × two FFmpeg threads. GPU stages use three image
  workers. CPU-only compaction/index stages may use six threads only after GPU
  loader/extractor processes exit.

- [ ] **Step 1: Write failing subprocess, order, and resource tests**

  Simulate success, 8→4→2 OOM recovery, OOM at one, non-OOM failure, missing
  output, bounded stderr, environment propagation without secrets, strict stage
  order, no extraction/model overlap, and no nested CPU oversubscription.
- [ ] **Step 2: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/data/custom_pipeline/test_stages.py -q
  ```
- [ ] **Step 3: Implement subprocess execution and resource environment**

  Use explicit argv and stage-scoped environment variables. Update state only
  after output validation; never edit native state JSON.
- [ ] **Step 4: Add disk admission before every material write stage**

  Record pre/post free bytes, active bytes, elapsed time, effective batch, and
  worker/thread settings in the batch report.
- [ ] **Step 5: Run stage and object CLI regressions**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_stages.py tests/scripts/test_detect_objects.py -q
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/stages.py \
    tests/data/custom_pipeline/test_stages.py
  git commit -m "feat(data): bound A6000 batch stage resources"
  ```

---

## Task 8: Atomically commit local batches and clean only ephemeral inputs

**Files:**

- Create: `src/hcmai/data/custom_pipeline/commit.py`
- Create: `tests/data/custom_pipeline/test_commit.py`

**Interfaces:**

- `build_batch_inventory(staging_root)` records relative paths, bytes, SHA-256,
  canonical digests, and three index inventories.
- `validate_local_batch(batch_id, video_ids, staging_root, inventory)` requires
  every specialist/vector mapping and checksum-loads all three indexes.
- `commit_local_batch(staging_root, final_batch_root, inventory)` writes
  `manifest.json` and `_SUCCESS.json`, then atomically renames the batch onto the
  same filesystem. A conflicting completed destination is accepted only when
  inventories match exactly.
- `cleanup_ephemeral_batch(...)` requires `BatchStage.committed`, preserves the
  committed durable keyframes/artifacts/indexes, and removes only inventoried
  OCR scratch, native links/staging, and source MP4s before advancing to
  `ephemeral_cleaned`.

- [ ] **Step 1: Write failing local commit tests**

  Cover payload-before-marker order, missing/corrupt payload, index load failure,
  atomic visibility, identical resume, conflicting destination, interruption
  before marker, and deterministic inventory order.
- [ ] **Step 2: Write failing cleanup-safety tests**

  Prove cleanup is forbidden before commit, preserves every final JPEG/Parquet/
  NumPy/index file, removes exact ephemeral paths after commit, is idempotent,
  retains MP4s after failure, and cannot escape archive/batch/native roots.
- [ ] **Step 3: Run tests and confirm failures**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/data/custom_pipeline/test_commit.py -q
  ```
- [ ] **Step 4: Implement atomic local commit and exact cleanup**

  Use safe relative inventories and existing atomic-write helpers. Never call a
  broad recursive delete on repository, `data`, `artifacts`,
  `artifacts_legacy`, or the full run root.
- [ ] **Step 5: Run commit/state regressions**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_commit.py \
    tests/data/custom_pipeline/test_state.py \
    tests/data/test_custom_state.py -q
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/commit.py \
    tests/data/custom_pipeline/test_commit.py
  git commit -m "feat(data): commit local batches before ephemeral cleanup"
  ```

---

## Task 9: Compact local batches into global corpus tables and indexes

**Files:**

- Create: `src/hcmai/data/custom_pipeline/finalize.py`
- Create: `tests/data/custom_pipeline/test_finalize.py`
- Modify: `src/hcmai/retrieval/retriever/dense/index.py`
- Modify: `src/hcmai/retrieval/retriever/segment/index.py`
- Modify: `tests/retrieval/test_context_index.py`
- Modify: `tests/retrieval/test_visual_embedding_resume.py`
- Modify: `tests/retrieval/test_segment_dense_index.py`

**Interfaces:**

- `compact_frame_metadata(batch_manifests, output)` validates retained local
  image paths without loading images.
- `compact_specialist_shards(kind, shard_paths, output)` streams deterministic
  Caption/OCR/Object/FrameContext Parquet.
- `compact_batch_embeddings(...)` validates batch markers, lineage, dimensions,
  checksums, and non-overlap before memory-mapped concatenation.
- `build_dense_index_from_precomputed(...)` and
  `build_segment_index_from_precomputed(...)` build/checksum-load global indexes
  without any encoder call.
- `finalize_corpus(context)` requires all frozen archives cleaned and writes
  corpus tables, global visual/context/ASR indexes, and `finalize_report.json`.

- [ ] **Step 1: Write failing deterministic compaction tests**

  Supply batches out of order. Assert stable identity order, exact counts,
  retained-image existence checks, empty child-table support, duplicate/foreign
  rejection, finite/dimension validation, and contiguous global mappings.
- [ ] **Step 2: Write failing global-index tests**

  Assert all three indexes load, retain canonical/segment identity and lineage,
  reject a missing/overlapping/corrupt batch, and perform no model inference.
- [ ] **Step 3: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_finalize.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_segment_dense_index.py -q
  ```
- [ ] **Step 4: Implement streaming Parquet and memory-mapped vector compaction**

  Process one batch/video at a time. Preallocate final arrays from validated
  counts and use at most six CPU threads during index construction. Do not load
  the full 470k-frame corpus in pandas or byte-merge FAISS files.
- [ ] **Step 5: Implement final validation report**

  Record complete archive/batch/video coverage, canonical counts, model/config/
  ASR lineage, elapsed stage totals, effective settings, peak resources, and
  exact global index load results.
- [ ] **Step 6: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_finalize.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_segment_dense_index.py \
    tests/retrieval/test_asr_segment_retriever.py -q
  ```
- [ ] **Step 7: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/finalize.py \
    src/hcmai/retrieval/retriever/dense/index.py \
    src/hcmai/retrieval/retriever/segment/index.py \
    tests/data/custom_pipeline/test_finalize.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_segment_dense_index.py
  git commit -m "feat(data): finalize local custom corpus indexes"
  ```

---

## Task 10: Compose preflight, rolling archives, status, and finalization

**Files:**

- Create: `src/hcmai/data/custom_pipeline/runner.py`
- Create: `tests/data/custom_pipeline/test_runner.py`
- Modify: `src/hcmai/data/ingestion/custom_enrichment.py`

**Interfaces:**

- `preflight_pipeline(options)` validates native/FFmpeg/curl, complete archive
  plan and work window, media-info, run identity, local roots, real disk,
  A6000 visibility, exactly six configured CPUs, model caches/revisions, and ASR
  lineage. It performs no archive download.
- `process_archive(options, archive_entry)` resumes one URL, inventories every
  member, processes groups of at most eight through local batch commit/cleanup,
  and cleans the archive only after exact member coverage.
- `pipeline_status(options)` is read-only and reports local stage counts,
  failures, retained/active/free bytes, current archive/batch, and recommended
  next archive offset.
- `finalize_pipeline(options)` requires full-plan local coverage and delegates
  to Task 9. Intermediate windows exit successfully without finalization.

- [ ] **Step 1: Write failing orchestration tests with fake stages/native code**

  Verify one active archive, no yt-dlp/ASR inference/cloud command, groups
  `[8, ..., remainder]`, strict stage order through three indexes, completed
  batch skip, partial resume, MP4 preservation on failure, exact cleanup after
  local commit, disk remeasurement, and full-plan-only finalization.

  Cover offset 2 + limit 3, omitted limit, cleaned overlap, gap rejection, and
  proof that limit never truncates an archive’s videos.
- [ ] **Step 2: Add interruption tests at every durable boundary**

  Parameterize extraction, each specialist, OCR scratch cleanup, embeddings,
  each index, local marker, atomic rename, and ephemeral cleanup. Resume must
  preserve identity and restart from the last validated state.
- [ ] **Step 3: Run focused tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/data/custom_pipeline/test_runner.py -q
  ```
- [ ] **Step 4: Implement preflight and rolling composition**

  Reuse `scripts/extract_custom_keyframes.py` and existing stage package APIs.
  Keep the runner explicit and sequential; do not create a generic workflow
  framework.
- [ ] **Step 5: Implement machine-readable local status**

  Intermediate windows report `complete_window=true`,
  `complete_corpus=false`, and `recommended_next_offset`. Never report a stage
  complete from file existence alone.
- [ ] **Step 6: Run orchestration/ingestion regressions**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline/test_runner.py \
    tests/data/test_custom_enrichment.py \
    tests/data/test_custom_frames.py \
    tests/scripts/test_custom_extraction_cli.py -q
  ```
- [ ] **Step 7: Commit**

  ```bash
  git add src/hcmai/data/custom_pipeline/runner.py \
    src/hcmai/data/ingestion/custom_enrichment.py \
    tests/data/custom_pipeline/test_runner.py
  git commit -m "feat(data): orchestrate local archive preparation batches"
  ```

---

## Task 11: Replace the monolithic custom CLI with local subcommands

**Files:**

- Modify: `scripts/prepare_custom_pipeline.py`
- Modify: `tests/scripts/test_custom_pipeline.py`
- Modify: `scripts/README.md`

**CLI:**

```text
prepare_custom_pipeline.py preflight [shared] --archive-url URL ... --offset N [--limit K]
prepare_custom_pipeline.py process-archive [shared] --archive-url URL
prepare_custom_pipeline.py status [shared]
prepare_custom_pipeline.py finalize [shared]
```

Shared options include config, media-info, run/output roots, native executable,
version, source, frame-store ID, transcript/index roots, and JSON report. There
are no video selectors, yt-dlp options, cloud destination options, or
externally supplied source-root shortcuts.

- [ ] **Step 1: Rewrite CLI parser/dispatch tests**

  Cover all subcommands, repeated ordered URLs, default/explicit windows,
  invalid offset/limit, archive-count semantics, unknown/out-of-order URL,
  local roots, JSON status, exit codes, no video/cloud/yt-dlp options, and
  bounded native/archive errors.
- [ ] **Step 2: Run failing CLI tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/scripts/test_custom_pipeline.py -q
  ```
- [ ] **Step 3: Reduce the script to parsing, dispatch, and error shaping**

  Keep model, archive, state, disk, commit, and compaction logic in package
  modules. Preserve secure media-info ZIP extraction.
- [ ] **Step 4: Update local diagnostic documentation**

  Include preflight-only, one-archive window, interrupted resume, adjacent
  window, status, and full-plan finalize examples.
- [ ] **Step 5: Run tests and compile**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/scripts/test_custom_pipeline.py tests/data/custom_pipeline -q
  PYTHONPATH=.:src aic/bin/python -m compileall -q \
    scripts/prepare_custom_pipeline.py src/hcmai/data/custom_pipeline
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add scripts/prepare_custom_pipeline.py scripts/README.md \
    tests/scripts/test_custom_pipeline.py src/hcmai/data/custom_pipeline
  git commit -m "refactor(scripts): expose local resumable pipeline commands"
  ```

---

## Task 12: Rewrite the runbook for A6000, six CPUs, and local-only output

**Files:**

- Modify: `docs/runbooks/bootstrap_and_run_custom_pipeline.sh`
- Create: `tests/runbooks/test_custom_pipeline_runbook.py`
- Modify: `docs/superpowers/plans/2026-08-24-cpp-ffmpeg-deterministic-1fps-extraction.md`

**Runbook contract:**

- Defaults: `RUN_ROOT=runs/custom-raw1fps-v1`,
  `OUTPUT_ROOT=artifacts/custom-raw1fps-v1`, `MIN_FREE_GIB=15`,
  `MAX_ACTIVE_GIB=30`, `MAX_VIDEOS_PER_BATCH=8`,
  `EXTRACTOR_PROCESSES=2`, `FFMPEG_THREADS_PER_PROCESS=2`,
  `IMAGE_WORKERS=3`, `CPU_ONLY_THREADS=6`.
- The existing `URLS=(...)` array remains the complete ordered source plan.
- `ARCHIVE_OFFSET=0` and optional `ARCHIVE_LIMIT` select only ZIP positions.
- Every invocation passes all URLs to preflight and loops only its window.
- `PREFLIGHT_ONLY=1` performs no video download.
- No cloud environment variable is required or read.
- No ZIP accumulation, yt-dlp, video-level limit, or permanent model gateway.

- [ ] **Step 1: Add runbook contract tests**

  Use stub executables to prove A6000/six-CPU preflight happens before video
  download, complete URL plan forwarding, offset 2 + limit 3 URL calls, failure
  stops the next archive, no cloud/yt-dlp command, one active archive, local
  batch cleanup ordering, and conditional full-plan finalization.
- [ ] **Step 2: Run failing tests and Bash syntax check**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest tests/runbooks/test_custom_pipeline_runbook.py -q
  bash -n docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  ```
- [ ] **Step 3: Implement the archive-window loop**

  Build/test C++, prepare Python, warm model caches, validate `nvidia-smi` and
  CPU count/config, bootstrap media-info, pass all URLs to preflight, then call
  one `process-archive` per window URL. Run status and finalize only when full
  coverage is complete.
- [ ] **Step 4: Remove obsolete runbook behavior**

  Remove download-all/unzip-all paths, `ZIP_LIMIT`, video `LIMIT`/`VIDEO_ID`,
  cloud env checks, upload steps, and cleanup of whole source/artifact roots.
- [ ] **Step 5: Run runbook/native gates**

  ```bash
  bash -n docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  PYTHONPATH=.:src aic/bin/python -m pytest tests/runbooks/test_custom_pipeline_runbook.py -q
  cmake --build build/keyframes_extraction --parallel
  ctest --test-dir build/keyframes_extraction --output-on-failure
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add docs/runbooks/bootstrap_and_run_custom_pipeline.sh \
    tests/runbooks/test_custom_pipeline_runbook.py \
    docs/superpowers/plans/2026-08-24-cpp-ffmpeg-deterministic-1fps-extraction.md
  git commit -m "docs(runbook): run local A6000 archive batches"
  ```

---

## Task 13: Remove stale orchestration references and document local ownership

**Files:**

- Modify: `src/hcmai/data/README.md`
- Modify: `src/hcmai/data/WORKFLOW.md`
- Modify: `scripts/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Inventory active references**

  ```bash
  rg -n "prepare_custom_pipeline|prepare_transcripts.py|yt-dlp|watch_url|S3_BUCKET_NAME|S3_PREFIX|ZIP_LIMIT|VIDEO_ID" \
    src scripts tests docs configs
  ```

  Classify diagnostic compatibility surfaces separately from the active custom
  orchestration path. Do not delete useful standalone stage CLIs.
- [ ] **Step 2: Update ownership documentation**

  Document archive windows, groups of eight, local batch markers, retained
  durable keyframes, exact ephemeral cleanup, reused ASR, A6000/6-vCPU limits,
  and local finalization. State that manual backup is outside pipeline state.
- [ ] **Step 3: Remove stale active-path references**

  Remove transcript generation, yt-dlp/watch acquisition, download-all archive
  accumulation, cloud publication, and video-level selection from active custom
  docs/config. Preserve teammate object detection and all data/artifact roots.
- [ ] **Step 4: Run reference, import, and diff checks**

  ```bash
  rg -n "yt-dlp|yt_dlp|watch_url|S3_BUCKET_NAME|S3_PREFIX" \
    src/hcmai/data/custom_pipeline scripts/prepare_custom_pipeline.py \
    docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai scripts
  git diff --check
  ```

  The reference search must find no active custom-pipeline invocation or cloud
  requirement. Historical migration notes may still name removed fields.
- [ ] **Step 5: Run focused data/script tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline \
    tests/scripts/test_custom_pipeline.py \
    tests/scripts/test_custom_extraction_cli.py \
    tests/scripts/test_detect_objects.py -q
  ```
- [ ] **Step 6: Commit**

  ```bash
  git add src/hcmai/data/README.md src/hcmai/data/WORKFLOW.md \
    scripts/README.md .gitignore
  git commit -m "docs(data): document local custom corpus lifecycle"
  ```

---

## Task 14: Run local release gates and record the baseline

**Files:**

- Create: `docs/runbooks/custom_pipeline_validation.md`
- Modify: `docs/superpowers/plans/2026-08-28-a6000-100gb-custom-pipeline.md`

- [ ] **Step 1: Run all focused custom-pipeline tests**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/custom_pipeline \
    tests/scripts/test_custom_pipeline.py \
    tests/runbooks/test_custom_pipeline_runbook.py -q
  ```
- [ ] **Step 2: Run adjacent data/enrichment/retrieval regressions**

  ```bash
  PYTHONPATH=.:src aic/bin/python -m pytest \
    tests/data/test_custom_manifest.py \
    tests/data/test_custom_frames.py \
    tests/data/test_custom_enrichment.py \
    tests/data/test_custom_state.py \
    tests/data/enrichment \
    tests/retrieval/test_visual_embedding_resume.py \
    tests/retrieval/test_context_index.py \
    tests/retrieval/test_segment_dense_index.py \
    tests/retrieval/test_asr_segment_retriever.py -q
  ```
- [ ] **Step 3: Run native/static gates**

  ```bash
  cmake --build build/keyframes_extraction --parallel
  ctest --test-dir build/keyframes_extraction --output-on-failure
  PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai scripts tests
  bash -n docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  git diff --check
  ```
- [ ] **Step 4: Run a synthetic local end-to-end fixture**

  Exercise extraction → Caption/OCR/Objects fixtures → FrameContext → embeddings
  → three batch indexes → local commit → ephemeral cleanup → global compaction.
  Interrupt once before local marker and once after commit but before cleanup.
- [ ] **Step 5: Record exact results**

  Save commands, pass counts, elapsed time, code revision, fixture artifact
  counts, and any environment skip reason in
  `docs/runbooks/custom_pipeline_validation.md`. Check off only measured steps.
- [ ] **Step 6: Commit**

  ```bash
  git add docs/runbooks/custom_pipeline_validation.md \
    docs/superpowers/plans/2026-08-28-a6000-100gb-custom-pipeline.md
  git commit -m "test(data): record local pipeline release gates"
  ```

---

## Task 15: Execute and tune the measured A6000 pilot

**Files:**

- Runtime: `runs/<version>/reports/a6000_first_archive.json`
- Runtime: `runs/<version>/reports/a6000_window_continuation.json`
- Modify: `configs/prepare.yaml`
- Modify: `docs/runbooks/custom_pipeline_validation.md`
- Modify: `docs/superpowers/plans/2026-08-28-a6000-100gb-custom-pipeline.md`

- [ ] **Step 1: Run non-downloading preflight on the target VM**

  ```bash
  PREFLIGHT_ONLY=1 ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  ```

  Confirm A6000 visibility, configured six-CPU profile, model revisions/cache,
  post-warmup free disk, ASR lineage/coverage, and native CTest. No video archive
  is downloaded.
- [ ] **Step 2: Run the first complete archive**

  ```bash
  VERSION=custom-raw1fps-a6000-pilot \
    RUN_ROOT=runs/custom-raw1fps-a6000-pilot \
    OUTPUT_ROOT=artifacts/custom-raw1fps-a6000-pilot \
    ARCHIVE_OFFSET=0 ARCHIVE_LIMIT=1 \
      ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  ```

  Verify all archive videos, groups of at most eight, manual frame samples,
  exact `frame_idx`, specialist coverage, three indexes per batch, retained
  durable keyframes, removed OCR scratch/source MP4s, and one interrupted resume.
- [ ] **Step 3: Measure and tune one variable at a time**

  Record frame throughput, p50/p95 model-batch latency, GPU utilization, peak
  VRAM, CPU utilization, RAM, OOM history, stage batch, worker/thread settings,
  active bytes, retained bytes, and minimum free bytes. Change only one setting
  per run and require identical artifact identity/counts before accepting it.
- [ ] **Step 4: Prove adjacent-window resume**

  ```bash
  VERSION=custom-raw1fps-a6000-pilot \
    RUN_ROOT=runs/custom-raw1fps-a6000-pilot \
    OUTPUT_ROOT=artifacts/custom-raw1fps-a6000-pilot \
    ARCHIVE_OFFSET=1 ARCHIVE_LIMIT=1 \
      ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  ```

  Confirm the same run identity, only archive position 1 acquired, status next
  offset 2, cleaned overlap idempotency, and deliberate offset-3 gap rejection.
- [ ] **Step 5: Decide whether retained keyframes fit 100 GB**

  Use measured retained-byte growth from both archives to project the full
  corpus. If the 15 GiB reserve would be violated, stop: do not silently lower
  JPEG quality or delete final keyframes. Record that manual batch offload or a
  larger disk is required before Task 16.
- [ ] **Step 6: Commit only measured configuration/documentation changes**

  ```bash
  git add configs/prepare.yaml docs/runbooks/custom_pipeline_validation.md \
    docs/superpowers/plans/2026-08-28-a6000-100gb-custom-pipeline.md
  git commit -m "perf(data): record measured A6000 pipeline settings"
  ```

---

## Task 16: Run, finalize, and verify the complete local corpus

**Files:**

- Runtime: `runs/<version>/state/`
- Runtime: `runs/<version>/reports/`
- Runtime: `artifacts/<version>/`
- Modify after completion: `docs/runbooks/custom_pipeline_validation.md`
- Modify after completion:
  `docs/superpowers/plans/2026-08-28-a6000-100gb-custom-pipeline.md`

- [ ] **Step 1: Freeze the full-run identity**

  Record complete archive-plan digest, media-info digest, config fingerprint,
  model revisions, ASR fingerprints, code revision, local roots, and measured
  resource settings. Use a new immutable version, not the pilot version.
- [ ] **Step 2: Process contiguous archive windows**

  Reuse the same version/run/output roots. For example:

  ```bash
  VERSION=custom-raw1fps-v1 RUN_ROOT=runs/custom-raw1fps-v1 \
    OUTPUT_ROOT=artifacts/custom-raw1fps-v1 \
    ARCHIVE_OFFSET=0 ARCHIVE_LIMIT=3 \
      ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh

  VERSION=custom-raw1fps-v1 RUN_ROOT=runs/custom-raw1fps-v1 \
    OUTPUT_ROOT=artifacts/custom-raw1fps-v1 \
    ARCHIVE_OFFSET=3 ARCHIVE_LIMIT=3 \
      ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh
  ```

  Continue from status’s recommended next offset. Never manually remove active
  files that are owned by an incomplete batch.
- [ ] **Step 3: Resolve failures without silent corpus reduction**

  Retry recoverable archive/model failures. A permanently unavailable member
  blocks finalization. A reduced corpus requires a new version and explicit
  archive plan.
- [ ] **Step 4: Finalize and validate exact coverage**

  Require no missing/duplicate canonical IDs, exact specialist rows, valid child
  references, exact visual/context mappings, exact ASR segments, retained image
  paths, and checksum-loadable visual/context/ASR global indexes.
- [ ] **Step 5: Record operational metrics**

  Record archive order/count, work-window history, group sizes, total videos/
  frames, stage throughput/time, effective batches/workers, peak disk/RAM/VRAM,
  minimum free disk, retained bytes, retries, and failures.
- [ ] **Step 6: Mark completion only from the final local report**

  Update Task 15/16 checkboxes from actual reports. Local artifact presence or a
  running process is not completion evidence.
- [ ] **Step 7: Commit the validation record**

  ```bash
  git add docs/runbooks/custom_pipeline_validation.md \
    docs/superpowers/plans/2026-08-28-a6000-100gb-custom-pipeline.md
  git commit -m "docs(data): record complete local custom corpus"
  ```

---

## Completion Definition

Implementation is complete when Tasks 1–14 pass locally. Operational rollout
is complete when Tasks 15–16 have reports proving:

- the full frozen archive plan is represented without skipped videos;
- archive offset/limit counts ZIPs, contiguous windows resume one identity, and
  archive members are always processed in groups of at most eight;
- only one archive is active and no yt-dlp/watch-URL acquisition occurs;
- canonical `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms` are exact;
- Caption, OCR, Objects, FrameContext, visual/context embeddings, and reused ASR
  remain independently validated;
- every batch has loadable visual, context, and ASR indexes before cleanup;
- ZIP, MP4, OCR scratch, and native staging cleanup occurs only after an atomic
  local batch commit;
- durable keyframes and final local artifacts are retained;
- the 15 GiB free reserve and 30 GiB active cap are respected;
- A6000 model batches and six-vCPU worker limits are measured and recorded;
- interruptions resume from durable local boundaries;
- global visual, context, and ASR indexes checksum-load without model reruns;
- final counts, resource peaks, throughput, and known failures are recorded;
- no automated cloud publication or remote state is part of the pipeline.
