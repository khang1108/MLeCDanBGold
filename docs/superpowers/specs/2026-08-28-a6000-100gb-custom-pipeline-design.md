# A6000 / 100 GB Local Custom Corpus Pipeline Design

**Date:** 2026-08-28

**Status:** Approved direction awaiting implementation

**Scope:** Local offline preparation from organizer video ZIPs through
deterministic frames, specialist evidence, embeddings, and loadable retrieval
indexes. Automated cloud publication is intentionally outside this design.

## 1. Goal

Run the complete custom corpus preparation pipeline on one NVIDIA A6000 machine
with a 100 GB local disk and 6 vCPUs:

```text
media-info ZIP from the runbook
  -> fixed ordered organizer video ZIP plan
  -> zero-based archive offset/limit work window
  -> safe download and unzip of exactly one archive
  -> canonical groups of at most 8 videos
  -> native C++/FFmpeg deterministic 1 FPS extraction
  -> Caption + OCR + Objects
  -> deterministic FrameContext
  -> visual + context embeddings
  -> reused ASR-vector subset
  -> visual + context + ASR indexes for the batch
  -> atomic local batch commit and ephemeral cleanup
  -> deterministic corpus compaction and global indexes
```

The pipeline writes final local artifacts only. The operator may manually copy
those outputs to S3 or another backup target, but the pipeline has no cloud
credentials, upload stage, remote checkpoint, or remote cleanup dependency.

Existing ASR remains valid because it is video/timeline-native. The custom run
validates and subsets existing transcripts and vectors; it never regenerates or
re-embeds ASR.

The current 1 FPS estimate is 470,428 frames. This is a planning number, not a
coverage override. Native extraction manifests and every video in the complete
frozen archive plan determine accepted corpus coverage.

## 2. Current-State Gap

The repository already contains the native extractor and major enrichment and
indexing stages. The current composition is not yet suitable for this machine:

- the runbook downloads and extracts multiple archives before processing;
- the Python command extracts too much of the corpus before enrichment;
- durable and OCR images accumulate through corpus-wide indexing;
- ASR is regenerated in the custom orchestration path;
- batch sizes and CPU worker counts are not measured A6000/6-vCPU settings;
- state does not represent one complete eight-video batch through three
  loadable indexes;
- cleanup is corpus-wide rather than tied to a locally validated batch.

The change is orchestration, state, local storage, and bounded execution. It
does not change specialist-model semantics, the sampling algorithm, or the
competition frame coordinate.

## 3. Design Principles

1. **Process one archive at a time.** Never retain the next ZIP while the
   current archive has an incomplete batch.
2. **Process complete groups of at most eight videos.** `limit` never truncates
   videos inside an archive.
3. **Bound measured bytes.** Video count is a scheduling ceiling, not a disk
   guarantee.
4. **Commit locally before cleanup.** ZIP, MP4, OCR scratch, and native staging
   are removed only after the batch artifacts and all three indexes validate.
5. **Retain required final data.** Durable keyframes, specialist evidence,
   embeddings, mappings, manifests, and indexes remain under the versioned
   artifact root. OCR-resolution scratch frames do not.
6. **Run one GPU-heavy stage at a time.** Model processes are stage-scoped so
   VRAM is returned before the next model starts.
7. **Use the six CPUs deliberately.** Extraction, image decoding, orchestration,
   and index construction have explicit, non-oversubscribed worker limits.
8. **Preserve canonical identity.** `video_id`, `frame_id`, `frame_idx`, and
   `timestamp_ms` survive every table, vector mapping, and index.
9. **Keep specialist evidence separate.** FrameContext is derived from Caption,
   OCR, and Objects and does not replace them or contain ASR.
10. **Resume only validated state.** A file’s presence alone never proves a
    stage or batch is complete.

## 4. Resource Envelope

The initial operating profile is:

| Resource or guardrail | Initial value | Rationale |
|---|---:|---|
| GPU | NVIDIA A6000 | One GPU-heavy stage at a time |
| Local filesystem | 100 GB | Includes environment, caches, retained outputs, and active work |
| Logical CPUs | 6 | Avoid nested thread-pool oversubscription |
| Minimum free reserve | 15 GiB | Preserve VM operability and temporary-write headroom |
| Maximum active work tree | 30 GiB | Bound current ZIP, MP4s, frames, and scratch |
| Videos per pipeline batch | 8 | GPU batching ceiling; disk admission remains authoritative |
| Parallel FFmpeg extractors | 2 | Two videos concurrently without consuming all CPUs |
| FFmpeg threads per extractor | 2 | Four decode threads total |
| GPU-stage image workers | 3 | Leave CPU capacity for the main process and filesystem |
| Image prefetch depth | 2 model batches | Bound decoded-image RAM |
| PyArrow/FAISS CPU threads | 6 in CPU-only stages | Use all CPUs only when no extractor/model loader is active |

Initial model batch hypotheses are Caption 8, OCR 32, Objects 32, visual 128,
and context 128. Each stage halves its batch on a recognized CUDA OOM down to
one and records the effective value. These numbers are PROPOSED until measured
on the A6000 pilot.

The disk admission invariant is:

```text
free_bytes - estimated_next_write_bytes >= 15 GiB
and
active_working_set_bytes + estimated_next_write_bytes <= 30 GiB
```

Both retained artifacts and active data count toward real filesystem free
space. Only `runs/<version>/active/` counts toward the 30 GiB active cap.
Model caches are warmed before corpus processing so their disk growth is visible
to preflight.

The runner measures unique `(st_dev, st_ino)` pairs so same-filesystem hard
links do not inflate usage. It refuses cross-device copy fallback. ZIP declared
sizes are security ceilings; actual free/active bytes remain authoritative.

## 5. Archive Work Windows

The runbook’s ordered `URLS=(...)` array is the sole archive plan. Every run
invocation passes the complete array to preflight and selects only an
operational window:

```text
window = archive_plan[offset:]                    # limit omitted
window = archive_plan[offset : offset + limit]    # limit supplied
```

- `offset` defaults to zero and is a zero-based ZIP position.
- `limit` is an optional positive number of ZIP files.
- `offset=2, limit=3` selects archive positions 2, 3, and 4.
- Neither value selects or limits videos.
- Work-window values are operational history, not artifact lineage.
- Starting at offset `N` requires archives `0..N-1` to be `cleaned`.
- Replaying an already-cleaned overlapping window is idempotent.
- A gap is rejected before downloading anything.

The complete ordered archive-plan digest, version, media-info digest, frame
store identity, artifact configuration, model revisions, and ASR lineage form
the immutable run identity.

## 6. Rolling Batch Lifecycle

### 6.1 Archive lifecycle

For each work-window URL:

1. derive a safe stable archive ID from the ZIP filename;
2. resume download into an archive-scoped `.part` file;
3. validate paths, links, member counts, sizes, duplicate IDs, and MP4 naming;
4. extract atomically into the current archive directory;
5. inventory every nested `Lxx/Lxx_Vnnn.mp4` member;
6. validate every member against media-info and reusable ASR coverage;
7. commit `archive_manifest.json`, then delete the ZIP;
8. sort all members by canonical `video_id` and partition them into groups of
   eight plus a final remainder;
9. process each group through the complete batch lifecycle;
10. delete each source MP4 only after its batch is locally committed;
11. remove the empty extracted/archive directory and mark the archive cleaned;
12. continue to the next URL.

### 6.2 Per-batch stages

For each group of at most eight videos:

1. create same-filesystem hard links in the native source root;
2. extract durable JPEGs and higher-resolution OCR scratch JPEGs;
3. validate native metadata, timestamps, frame identities, and `frame_idx`;
4. atomically move validated durable JPEGs into the batch’s artifact staging
   directory on the final filesystem so later cleanup never removes them;
5. run Caption over durable JPEGs;
6. run OCR over OCR scratch JPEGs, validate OCR tables, then delete only that
   batch’s OCR scratch;
7. run Objects over durable JPEGs;
8. build deterministic FrameContext on CPU;
9. build visual embedding vectors/mapping from durable JPEGs;
10. build context embedding vectors/mapping from FrameContext;
11. subset persisted ASR vectors/mapping for exactly the batch video IDs;
12. build and checksum-load visual `DenseIndex`, context `DenseIndex`, and ASR
    `SegmentDenseIndex` for the batch;
13. validate exact specialist coverage, vector counts, canonical mappings,
    model/config lineage, and all three indexes;
14. write the local batch manifest and `_SUCCESS.json` atomically;
15. remove exact native source links, native staging, and source MP4s;
16. retain durable keyframes and compact batch artifacts/indexes.

The batch is the cleanup boundary. A video may resume independently inside the
batch, but no source MP4 or native staging for that batch is deleted before the
batch marker validates.

### 6.3 Failure behavior

- Stage outputs use sibling temporary paths and atomic rename.
- Resume validates schema, model/config fingerprint, canonical identity digest,
  row/vector counts, and file checksums.
- A mismatch invalidates only that stage and its descendants in the batch.
- Recognized CUDA OOM halves the model batch size; unrelated failures are not
  mislabeled or retried as OOM.
- A failed local batch preserves its archive MP4s and validated intermediate
  outputs for resume.
- Cleanup uses exact inventoried paths and never recursively targets `data/`,
  `artifacts/`, `artifacts_legacy/`, the repository root, or the full run root.

## 7. Local State Machine

Archive state:

```text
pending -> downloading -> downloaded -> extracted -> processing
        -> complete -> cleaned
```

Batch state:

```text
planned -> extracted -> artifacts_complete -> indexes_complete
        -> committed -> ephemeral_cleaned
```

Video state:

```text
pending -> source_ready -> extracted -> captioned -> ocr_complete
        -> objects_complete -> context_complete -> embeddings_complete
        -> local_complete
```

Failure metadata records the failed stage, bounded diagnostics, attempt count,
and timestamp while retaining the last successful state. State is one atomic
JSON file per archive, batch, and video; native C++ state remains separate.

## 8. Canonical Data Contracts

The native extractor remains authoritative for the selected timestamp and
computes:

```text
frame_idx = floor(ceil(avg_fps_of_video) * timestamp_ms / 1000)
```

Python recomputes this only as a rejection check. `frame_id` is the internal
join key, `frame_idx` is the competition coordinate, and keyframe order never
substitutes for either.

Every frame-native Caption, OCR-frame, Object-frame, FrameContext, visual
mapping, and context mapping table has exact canonical frame coverage. OCR
regions and object detections may have zero or more rows but cannot reference a
foreign frame. Repeated object labels remain repeated when multiplicity matters.
Raw and normalized OCR remain independently available.

ASR remains segment-native. The batch ASR index preserves `segment_id`,
`video_id`, `start_ms`, and `end_ms`; speech is not projected into FrameContext.

## 9. Local Artifact Layout

```text
runs/<version>/
├── input/                         # media-info and frozen archive plan
├── state/
│   ├── run.json
│   ├── archives/<archive_id>.json
│   ├── batches/<batch_id>.json
│   └── videos/<video_id>.json
├── active/
│   ├── archives/                  # at most one current archive
│   ├── native/                    # hard links, native staging/state
│   └── batch/                     # current stage scratch only
└── reports/

artifacts/<version>/
├── batches/<archive_id>/<batch_id>/
│   ├── videos/<video_id>/
│   │   ├── frames/images/*.jpg
│   │   ├── frames/frames.parquet
│   │   ├── enrichment/*.parquet
│   │   ├── embeddings/*.npy
│   │   ├── embeddings/*_mapping.parquet
│   │   └── native/manifest.json
│   ├── indexes/{visual,context,asr_segments}/
│   ├── manifest.json
│   └── _SUCCESS.json
├── corpus/*.parquet
├── indexes/{visual,context,asr_segments}/
└── reports/finalize_report.json
```

The batch marker contains ordered video IDs, canonical identity digests,
artifact/model lineage, file sizes, and SHA-256 inventories. It is local resume
state, not a cloud publication contract.

## 10. Finalization

Finalization starts only when every archive in the complete frozen plan is
cleaned and every batch marker validates. It:

1. streams frame and specialist Parquet shards in deterministic
   `(video_id, timestamp_ms, frame_id)` order;
2. validates retained durable-image paths without loading all images;
3. compacts visual/context/ASR vectors and mappings from batch outputs;
4. rejects missing, duplicate, foreign, non-finite, or dimension-incompatible
   vectors;
5. rewrites global `embedding_index` to `0..N-1`;
6. builds global visual/context `DenseIndex` and ASR `SegmentDenseIndex` from
   precomputed vectors without rerunning models or byte-merging FAISS files;
7. checksum-loads all three global indexes;
8. writes `finalize_report.json` atomically.

Intermediate archive windows exit successfully with local batch indexes and a
recommended next offset, but they cannot produce the final corpus report.

## 11. A6000 and 6-vCPU Execution Strategy

GPU stages are separate subprocesses in this strict order: Caption, OCR,
Objects, visual embedding, context embedding. CPU-only FrameContext and index
construction run between/after them as defined by the batch lifecycle.

Only one thread pool is authoritative at a time:

- extraction: two processes × two FFmpeg threads;
- GPU inference: three image-loader workers and no concurrent extractor;
- FrameContext/Parquet: bounded streaming, no pandas full-corpus load;
- FAISS/global vector compaction: up to six threads, no GPU model process;
- orchestration/checksum work: one main process, bounded queues.

The pilot records per-stage wall time, frames/s, p50/p95 batch latency, CPU/GPU
utilization, RAM/VRAM, OOM retries, effective model batch size, and peak disk.
No batch or worker value is called optimal before measurement.

## 12. CLI and Runbook

The Python CLI is a thin entry point:

```text
prepare_custom_pipeline.py preflight ... --archive-url URL ... --offset N [--limit K]
prepare_custom_pipeline.py process-archive ... --archive-url URL
prepare_custom_pipeline.py status ...
prepare_custom_pipeline.py finalize ...
```

Preflight receives the complete ordered URL plan. `process-archive` accepts one
URL in the active work window. `status` reports local state, bytes, current
archive/batch, and recommended next offset. `finalize` requires complete local
coverage.

The runbook no longer requires cloud environment variables. It builds/tests the
native extractor, prepares Python/model caches, validates the A6000 and six-CPU
profile, downloads media-info once, passes the full URL array to preflight, and
loops one work-window archive at a time. It calls finalization only after the
full plan is locally complete.

## 13. Validation Gates

### Gate A: deterministic local fixtures

- archive offset/limit and gap rejection;
- exact groups of eight plus final remainder;
- byte admission, hard-link accounting, and path-safe cleanup;
- canonical identity and `frame_idx` rejection;
- stage resume and OOM backoff;
- exact Caption/OCR/Object/FrameContext coverage;
- three loadable batch indexes from precomputed vectors;
- interruption before/after local batch commit;
- deterministic global compaction and three loadable global indexes.

### Gate B: first complete archive on A6000

Run `offset=0, limit=1`, process every archive member, inspect sampled durable
and OCR images, validate each batch, and record the resource report. Interrupt
one batch stage and prove resume.

### Gate C: adjacent-window continuation

Reuse the same version and roots with `offset=1, limit=1`. Confirm the immutable
run identity is unchanged, status recommends offset 2, a cleaned overlap is
idempotent, and a gap is rejected.

### Gate D: full corpus

Completion requires every video and batch from every frozen archive, exact
specialist/vector coverage, three loadable global indexes, retained durable
keyframes, at least 15 GiB free throughout, and a reproducible final report.

## 14. Non-goals

- automated S3 upload, synchronization, verification, or remote cleanup;
- changing the 1 FPS sampling algorithm or `frame_idx` formula;
- regenerating ASR;
- changing BTC-provided keyframes or the BTC preparation profile;
- serving-time enrichment or index rebuilding;
- simultaneous multi-model GPU scheduling;
- distributed orchestration or a generic workflow framework;
- deleting `data/`, `artifacts/`, or `artifacts_legacy/`.

Manual backup/synchronization of `artifacts/<version>/` is operator-owned and
occurs after or between pipeline windows without changing pipeline state.

## 15. Assumptions to Validate

- Retained durable JPEGs plus compact artifacts fit the 100 GB filesystem while
  the runner preserves a 15 GiB reserve. If the pilot disproves this, the
  operator must manually offload completed batch directories or provision more
  disk; the pipeline must not silently lower image quality or delete final
  keyframes.
- A6000 throughput and the proposed model batches are not yet measured.
- Two FFmpeg extractors with two threads each are faster than one extractor on
  the actual six-vCPU VM without causing I/O contention.
- Existing transcript and ASR-index coverage includes every archive video.
- Organizer archives individually satisfy the active-cap and free-reserve
  checks while compressed and extracted bytes briefly coexist.
