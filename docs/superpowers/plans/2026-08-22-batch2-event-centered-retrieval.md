# Batch-2 Event-Centered Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Batch-2 architecture jump: deterministic 1-FPS raw-video frames, validated competition coordinates, BTC/custom/fused visual retrieval, boundary-aware `SceneCandidate` localization, training-free late interaction, and a paper-ready ablation/evaluation harness.

**Architecture:** Reuse existing contracts instead of introducing a parallel event model. Raw-video observations are ordinary `FrameRecord` rows; query-conditioned temporal scenes remain `SceneCandidate`. BTC and custom FAISS indexes are merged inside one VISUAL-channel retriever, then the existing Context/ASR-segment fusion and `TemporalEvidenceCore` continue to operate on canonical frame IDs.

**Tech Stack:** Python, Pydantic, pandas/Parquet, PyAV, NumPy, FAISS, SigLIP2 adapter already in the repository, TransNetV2, EfficientGEBD, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-event-centered-zero-shot-retrieval-design.md`

## Global Constraints

- Reuse `FrameRecord`, `QueryUnit`, `FrameEvidence`, and `SceneCandidate`; do not create `FrameObservation`, `TemporalEvent`, or `EventStore` in this plan.
- `frame_id` remains internal immutable identity; retrieval/index positions never become identity.
- `FrameRecord.frame_idx` is computed offline for custom frames and remains the sole submission coordinate consumed by `official_frame_idx()`.
- BTC organizer `frame_idx` remains untouched for BTC rows.
- `keyframe_order=None` is valid for custom frames and must not block embedding/index construction.
- BTC/custom are two corpora of `RetrievalSource.VISUAL`; do not add new `RetrievalSource` enum values.
- Specialist evidence remains independent; no destructive flattening.
- ASR stays timeline-native; segment-to-frame projection uses whichever frame corpus is active for that ablation.
- Offline generation is explicit and resumable. Online search must never decode videos or build artifacts.
- The ~20 official benchmark queries are evaluation-only; they are not training data.
- Batch-2 ablations must use the same codepath: BTC/custom/fused, frame-only/scene, boundary off/on, max/log-mean-exp pooling.
- No broad legacy cleanup until the Batch-2 path is passing; only remove legacy code directly replaced by a task below.

---

## File Structure Locked by This Plan

### New focused implementation files

- `src/hcmai/data/preprocessing/submission.py` — pure custom-frame submission-coordinate mapping and BTC reverse-validation helpers. This is implementation logic, not a shared schema.
- `src/hcmai/data/preprocessing/boundaries.py` — build/query persisted soft boundary priors; Parquet is module-local artifact shape, not a new Pydantic contract.
- `src/hcmai/retrieval/retriever/dense/corpus.py` — one VISUAL retriever over one or two dense visual corpora, with one query encoding and deterministic same-channel fusion.

### Existing files intentionally evolved

- `src/hcmai/common/schemas/frame.py` — no new fields; tests only unless a nullable-load bug is exposed.
- `src/hcmai/data/preprocessing/config.py` — replace adaptive-selection settings with deterministic timeline/boundary settings used by the new producer.
- `src/hcmai/data/preprocessing/video.py` — preserve PyAV/PTS helpers; add deterministic 1-FPS streaming sampler.
- `src/hcmai/data/preprocessing/prepare.py` — replace query-independent candidate selection/dedup with single-pass 1-FPS materialization.
- `src/hcmai/data/preprocessing/s3.py` — keep S3 transport but point it at the new deterministic producer.
- `src/hcmai/data/preprocessing/__init__.py` — export the new canonical preprocessing API.
- `src/hcmai/data/pipeline.py` — expose timeline preparation instead of `prepare_adaptive` legacy naming.
- `src/hcmai/data/stores/frame.py` — add strict offline metadata merge helper; runtime lookup behavior stays unchanged.
- `src/hcmai/retrieval/embedding/artifacts.py` — make visual artifact construction generic for nullable `keyframe_order`.
- `scripts/build_retrieval_indexes.py` — separate BTC-only preflight from generic visual build; do not fork a custom builder.
- `src/hcmai/retrieval/retriever/pipeline.py` — compose the multi-corpus visual retriever with existing text/ASR retrieval.
- `src/hcmai/orchestration/setup.py` — load one or two visual indexes according to config.
- `src/hcmai/common/config.py` — add only execution/ablation configuration, not domain schemas.
- `src/hcmai/temporal/aligners/scene.py` — boundary-aware initial clustering plus soft adjacent-scene recombination.
- `src/hcmai/temporal/scoring.py` — configurable max vs log-mean-exp unit pooling.
- `src/hcmai/temporal/core.py` — inject boundary provider into the existing `ProgressiveSceneAligner`; no new temporal core.
- `src/hcmai/orchestration/workflows/kis.py` — deterministic representative-frame refinement and frame-only ablation path.
- `src/hcmai/retrieval/retriever/evaluation/benchmark.py` — extend the existing benchmark instead of adding a parallel evaluator.
- `scripts/build_benchmark.py` — make the existing CLI load the current app/retrieval configuration rather than its obsolete `models.embedding` schema.

### Legacy code removed only when replaced

- `src/hcmai/data/preprocessing/selection.py` — delete after the deterministic producer and its tests are green.
- Adaptive-selection-specific tests in `tests/preprocessing/test_frame_store.py` — replace with deterministic sampling/identity/resume tests; do not retain dead tests merely to preserve deleted behavior.

---

### Task 1: Authoritative Custom Submission-Frame Mapping

**Files:**
- Create: `src/hcmai/data/preprocessing/submission.py`
- Modify: `src/hcmai/data/preprocessing/__init__.py`
- Create: `scripts/validate_submission_mapping.py`
- Modify: `scripts/script_args.py`
- Test: `tests/preprocessing/test_submission_mapping.py`
- Test: `tests/data/test_btc_keyframe_map.py`

**Interfaces:**
- Consumes: BTC mapping `DataFrame` returned by `load_btc_keyframe_map()` and raw-video `timestamp_ms` / average FPS.
- Produces:
  - `normalize_submission_fps(source_fps: float) -> int`
  - `submission_frame_idx(timestamp_ms: int, source_fps: float, rule: str) -> int`
  - `validate_submission_rule(mapping: pd.DataFrame, rule: str) -> pd.DataFrame` returning only mismatch rows.
- Full-corpus timeline generation must receive an explicit validated rule (`floor`, `round`, or `ceil`); no hidden default is allowed in the CLI/config used for the full corpus.

- [ ] **Step 1: Write failing pure-function tests for FPS normalization and frame-index rules**

```python
# tests/preprocessing/test_submission_mapping.py
import pytest

from hcmai.data.preprocessing.submission import (
    normalize_submission_fps,
    submission_frame_idx,
)


def test_normalize_submission_fps_snaps_ntsc_to_30() -> None:
    assert normalize_submission_fps(29.97) == 30


def test_normalize_submission_fps_preserves_25() -> None:
    assert normalize_submission_fps(25.0) == 25


def test_normalize_submission_fps_rejects_unknown_rate() -> None:
    with pytest.raises(ValueError, match="25 or 30"):
        normalize_submission_fps(23.976)


def test_submission_frame_idx_uses_explicit_rule() -> None:
    assert submission_frame_idx(79, 25.0, "floor") == 1
    assert submission_frame_idx(79, 25.0, "round") == 2
    assert submission_frame_idx(79, 25.0, "ceil") == 2
```

- [ ] **Step 2: Run the tests and verify they fail because the mapping module does not exist**

Run:

```bash
PYTHONPATH=src python -m pytest tests/preprocessing/test_submission_mapping.py -v
```

Expected: import failure for `hcmai.data.preprocessing.submission`.

- [ ] **Step 3: Implement the minimal pure mapper**

```python
# src/hcmai/data/preprocessing/submission.py
from __future__ import annotations

import math

_ALLOWED_SUBMISSION_FPS = (25, 30)
_ALLOWED_RULES = {"floor", "round", "ceil"}


def normalize_submission_fps(source_fps: float) -> int:
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    nearest = min(_ALLOWED_SUBMISSION_FPS, key=lambda fps: abs(fps - source_fps))
    if abs(nearest - source_fps) > 0.2:
        raise ValueError("submission FPS must resolve to organizer FPS 25 or 30")
    return nearest


def submission_frame_idx(timestamp_ms: int, source_fps: float, rule: str) -> int:
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if rule not in _ALLOWED_RULES:
        raise ValueError("rule must be one of: floor, round, ceil")
    value = timestamp_ms * normalize_submission_fps(source_fps) / 1000.0
    if rule == "floor":
        return math.floor(value)
    if rule == "ceil":
        return math.ceil(value)
    return round(value)
```

- [ ] **Step 4: Add BTC reverse-validation tests**

```python
# tests/preprocessing/test_submission_mapping.py
import pandas as pd

from hcmai.data.preprocessing.submission import validate_submission_rule


def test_validate_submission_rule_returns_only_mismatches() -> None:
    mapping = pd.DataFrame([
        {"video_id": "V1", "pts_time": 1.001, "fps": 29.97, "frame_idx": 30},
        {"video_id": "V1", "pts_time": 2.002, "fps": 29.97, "frame_idx": 60},
    ])
    assert validate_submission_rule(mapping, "floor").empty
```

Implement validation by converting `pts_time` to milliseconds using the same rounding already used by `join_btc_mapping`, applying `submission_frame_idx`, and returning a copy of rows whose predicted coordinate differs from organizer `frame_idx`. Include `predicted_frame_idx` and `delta` columns in mismatches.

- [ ] **Step 5: Add the validation CLI**

The CLI must load organizer maps, evaluate all three rules, print exact match counts, and exit non-zero unless one rule has zero mismatches. It must never auto-pick a non-perfect rule.

```python
# scripts/validate_submission_mapping.py (core behavior)
mapping = load_btc_keyframe_map(Path(args.mapping_root))
reports = {
    rule: validate_submission_rule(mapping, rule)
    for rule in ("floor", "round", "ceil")
}
perfect = [rule for rule, mismatches in reports.items() if mismatches.empty]
args.output.mkdir(parents=True, exist_ok=True)
summary = {}
for rule, mismatches in reports.items():
    summary[rule] = {
        "matches": int(len(mapping) - len(mismatches)),
        "mismatches": int(len(mismatches)),
    }
    if not mismatches.empty:
        mismatches.to_csv(args.output / f"{rule}-mismatches.csv", index=False)
write_json(summary, args.output / "summary.json")
if len(perfect) != 1:
    return 1
print(perfect[0])
return 0
```

Use concrete CLI arguments in `scripts/script_args.py`:

```text
--mapping-root PATH
--output PATH
```

The output directory contains `summary.json` plus `<rule>-mismatches.csv` for non-empty reports.

- [ ] **Step 6: Run mapper tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/preprocessing/test_submission_mapping.py \
  tests/data/test_btc_keyframe_map.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/data/preprocessing/submission.py \
        src/hcmai/data/preprocessing/__init__.py \
        scripts/validate_submission_mapping.py scripts/script_args.py \
        tests/preprocessing/test_submission_mapping.py \
        tests/data/test_btc_keyframe_map.py
git commit -m "feat: validate custom submission frame coordinates"
```

---

### Task 2: Replace Adaptive Keyframe Selection with a Deterministic 1-FPS `FrameRecord` Producer

**Files:**
- Modify: `src/hcmai/data/preprocessing/config.py`
- Modify: `src/hcmai/data/preprocessing/video.py`
- Rewrite focused sections: `src/hcmai/data/preprocessing/prepare.py`
- Modify: `src/hcmai/data/preprocessing/s3.py`
- Modify: `src/hcmai/data/preprocessing/__init__.py`
- Modify: `src/hcmai/data/pipeline.py`
- Modify: `scripts/preprocess_videos.py`
- Delete after migration: `src/hcmai/data/preprocessing/selection.py`
- Modify: `configs/preparation.s3.yaml`
- Rewrite tests: `tests/preprocessing/test_frame_store.py`
- Modify tests: `tests/preprocessing/test_s3_frame_store.py`

**Interfaces:**
- Consumes: raw video path/S3-staged video, `PreprocessingConfig`, explicit validated `submission_rounding`.
- Produces: a resumable `frames.parquet` containing only existing `FrameRecord` fields and one image per deterministic 1-FPS sample.
- Public library entry point remains `prepare_frame_store(config, resume=True, limit=None) -> Path` to avoid proliferating builders; semantics/documentation change from adaptive keyframes to deterministic timeline.

- [ ] **Step 1: Replace adaptive-selection tests with deterministic sampling tests**

Write tests that prove:

```python
def test_uniform_sampler_emits_one_frame_per_second(tmp_path: Path) -> None:
    path = tmp_path / "three_seconds.mp4"
    _video(path, seconds=3, fps=10)
    samples = list(iter_uniform_source_frames(path, sample_fps=1.0))
    timestamps = [meta.timestamp_ms for meta, _frame, _fps in samples]
    assert timestamps == pytest.approx([0, 1000, 2000], abs=100)


def test_custom_frame_record_reuses_canonical_schema(tmp_path: Path) -> None:
    path = tmp_path / "one_second.mp4"
    _video(path, seconds=1, fps=25)
    meta, source_frame, source_fps = next(
        iter_uniform_source_frames(path, sample_fps=1.0)
    )
    image_path = Path("images/V1/000000000.jpg")
    record = _record_from_sample(
        video_id="V1",
        meta=meta,
        source_average_fps=source_fps,
        image_path=image_path,
        submission_rounding="round",
    )
    assert isinstance(record, FrameRecord)
    assert record.keyframe_order is None
    assert record.fps in {25.0, 30.0}
    assert record.pts is not None
    assert record.time_base is not None
    source_frame.close()


def test_prepare_decodes_each_video_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "V1.mp4"
    _video(path, seconds=2, fps=25)
    config = _config(
        tmp_path,
        videos_root=tmp_path,
        sampling_fps=1.0,
        submission_rounding="round",
    )
    real_open = av.open
    calls = 0

    def counting_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", counting_open)
    prepare_frame_store(config, resume=False)
    assert calls == 1
```

Also keep the existing atomic publication/resume/failure-recovery tests that remain relevant.

- [ ] **Step 2: Run the rewritten preprocessing tests and verify failures against old adaptive semantics**

```bash
PYTHONPATH=src python -m pytest tests/preprocessing/test_frame_store.py -v
```

Expected: FAIL where `select_candidates`/DINO behavior is still active and where the current producer decodes twice.

- [ ] **Step 3: Simplify `PreprocessingConfig` around the new producer**

Keep S3 transport, image quality, TransNet/GEBD model settings needed later for boundary generation, and worker settings. Add:

```python
sampling_fps: float = Field(default=1.0, gt=0, le=4.0)
submission_rounding: Literal["floor", "round", "ceil"]
```

Remove fields used only by deleted candidate selection/deduplication:

```text
dino_model
dino_revision
dino_dtype
dino_batch_size
motion_threshold
minimum_gap_ms
maximum_gap_ms
dedup_similarity
```

Do not remove TransNet/EfficientGEBD configuration in this task.

- [ ] **Step 4: Add a streaming uniform sampler to `video.py`**

Use `iter_source_frames()` as the decode authority. Add a generator that chooses the source frame nearest each deterministic target time without keeping decoded images after they are emitted.

Required interface:

```python
def iter_uniform_source_frames(
    path: Path,
    *,
    sample_fps: float,
) -> Iterator[tuple[FrameMeta, Any, float]]:
    """Yield (frame_meta, source_frame, source_average_fps)."""
```

Algorithm:

```text
target period = 1000 / sample_fps
keep previous decoded frame
when current timestamp crosses target_ms:
    choose previous/current with smaller |timestamp-target|
    emit exactly once for that target
    advance target_ms by period until it is ahead of current
never emit the same decode_index twice
```

Do not derive competition `frame_idx` inside `iter_source_frames`; `FrameMeta.frame_idx` becomes decode/native metadata only or is removed from the internal dataclass if no remaining consumer needs it.

- [ ] **Step 5: Rewrite `_materialize`/`_prepare_video` as a single-pass producer**

For every sampled frame:

```python
submission_fps = normalize_submission_fps(source_average_fps)
record = FrameRecord(
    frame_id=f"{video_id}_raw1fps_{meta.decode_index:09d}",
    video_id=video_id,
    frame_idx=submission_frame_idx(
        meta.timestamp_ms,
        source_average_fps,
        config.submission_rounding,
    ),
    keyframe_order=None,
    timestamp_ms=meta.timestamp_ms,
    fps=float(submission_fps),
    image_path=relative_image_path.as_posix(),
    width=meta.width,
    height=meta.height,
    pts=meta.pts,
    time_base=meta.time_base,
)
```

Leave old selector-derived `FrameRecord` fields at their schema defaults; do not populate `event_id`, `shot_id`, or `selection_reasons` for the 1-FPS corpus.

Save the image immediately while the PyAV frame is live. Keep checkpoint validation and atomic directory publication, but fingerprint only configuration that affects deterministic timeline outputs.

- [ ] **Step 6: Remove the old selector from the active path**

Delete imports/usages of:

```text
CandidateFrame
select_candidates
deduplicate
restore_maximum_gap
DinoEncoder used only for keyframe pruning
```

Once no runtime/test import remains:

```bash
git rm src/hcmai/data/preprocessing/selection.py
```

Keep `models.py` because TransNetV2/EfficientGEBD are reused in Task 6.

- [ ] **Step 7: Rename the public DataService method without keeping a second implementation**

Replace:

```python
DataService.prepare_adaptive(config_path, resume=True, limit=3)
```

with:

```python
DataService.prepare_timeline(config_path, resume=True, limit=3)
```

Update all repository callers. Do not keep `prepare_adaptive` as a permanent compatibility alias; git history is the rollback path.

- [ ] **Step 8: Run local and S3 preprocessing tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/preprocessing/test_frame_store.py \
  tests/preprocessing/test_s3_frame_store.py -v
```

Expected: PASS, including one-decode-per-video, resume, atomic publication, and `keyframe_order=None` behavior.

- [ ] **Step 9: Commit**

```bash
git add src/hcmai/data/preprocessing src/hcmai/data/pipeline.py \
        scripts/preprocess_videos.py configs/preparation.s3.yaml tests/preprocessing
git commit -m "feat: build deterministic one-fps frame timeline"
```

---

### Task 3: Compose BTC and Custom Rows into One Runtime Frame Authority

**Files:**
- Modify: `src/hcmai/data/stores/frame.py`
- Create: `tests/data/test_frame_store_merge.py`
- Create: `scripts/build_combined_frame_store.py`
- Modify: `scripts/script_args.py`

**Interfaces:**
- Consumes: BTC `frames.parquet`, custom 1-FPS `frames.parquet`, their asset-root prefixes relative to the runtime dataset root.
- Produces: `merge_frame_metadata(inputs, output_path) -> Path` writing a combined `frames.parquet` with the exact `FrameRecord` columns.
- Runtime `FrameStore` lookup API does not change.

- [ ] **Step 1: Write failing merge tests**

```python
def test_merge_frame_metadata_preserves_distinct_identity_and_duplicate_coordinates(tmp_path):
    btc = _frame("btc-f1", "V1", frame_idx=25, timestamp_ms=1000, image_path="keyframes/V1/1.jpg")
    custom = _frame("raw-f1", "V1", frame_idx=25, timestamp_ms=1000, image_path="custom_1fps/images/V1/1.jpg", keyframe_order=None)
    btc_path = tmp_path / "btc.parquet"
    custom_path = tmp_path / "custom.parquet"
    pd.DataFrame([btc.model_dump(mode="python")]).to_parquet(btc_path, index=False)
    pd.DataFrame([custom.model_dump(mode="python")]).to_parquet(custom_path, index=False)
    output = merge_frame_metadata(
        [(btc_path, None), (custom_path, None)],
        tmp_path / "combined.parquet",
    )
    store = FrameStore(output)
    assert len(store) == 2
    assert store.get("btc-f1").frame_idx == store.get("raw-f1").frame_idx == 25


def test_merge_frame_metadata_rejects_duplicate_frame_id(tmp_path):
    duplicate = _frame(
        "same-id", "V1", frame_idx=25, timestamp_ms=1000, image_path="a.jpg"
    )
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    pd.DataFrame([duplicate.model_dump(mode="python")]).to_parquet(left, index=False)
    pd.DataFrame([duplicate.model_dump(mode="python")]).to_parquet(right, index=False)
    with pytest.raises(ValueError, match="Duplicate frame_id|duplicate frame_id"):
        merge_frame_metadata([(left, None), (right, None)], tmp_path / "out.parquet")
```

- [ ] **Step 2: Run the tests and confirm the helper is missing**

```bash
PYTHONPATH=src python -m pytest tests/data/test_frame_store_merge.py -v
```

Expected: FAIL import/name resolution.

- [ ] **Step 3: Implement `merge_frame_metadata` in `data/stores/frame.py`**

Use `FrameRecord.model_fields` as the output column authority. Validate every input row by constructing a temporary `FrameStore` or `FrameRecord`; concatenate without deduplicating `(video_id, frame_idx)`; reject duplicate `frame_id` globally; write atomically.

Signature:

```python
def merge_frame_metadata(
    inputs: Sequence[tuple[Path, str | None]],
    output_path: Path,
) -> Path:
    """Merge frame stores; optional prefix rebases relative image/thumbnail paths."""
```

For a custom frame store rooted under `data/custom_1fps`, call with prefix `custom_1fps`; for BTC rows already relative to `data`, pass `None`.

- [ ] **Step 4: Add a thin CLI**

```bash
PYTHONPATH=src python scripts/build_combined_frame_store.py \
  --btc-frames artifacts/frame_store/btc/frames.parquet \
  --custom-frames data/custom_1fps/frames.parquet \
  --custom-prefix custom_1fps \
  --output artifacts/frame_store/combined/frames.parquet
```

The CLI delegates all validation to `merge_frame_metadata`; do not duplicate merge logic inside the script.

- [ ] **Step 5: Run merge and existing FrameStore tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/data/test_frame_store_merge.py \
  tests/test_frame_assets.py \
  tests/test_frame_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/data/stores/frame.py \
        scripts/build_combined_frame_store.py scripts/script_args.py \
        tests/data/test_frame_store_merge.py
git commit -m "feat: compose btc and custom frame metadata"
```

---

### Task 4: Generalize Existing Visual Embedding/Index Builders for `keyframe_order=None`

**Files:**
- Modify: `src/hcmai/retrieval/embedding/artifacts.py:36-43,118-140,248-266`
- Modify: `scripts/build_retrieval_indexes.py` BTC preflight/build sections around current `keyframe_order` checks
- Modify: `tests/retrieval/test_visual_embedding_resume.py`
- Modify: `tests/retrieval/test_context_index.py` only if generic mapping validation shares helpers
- Modify: `tests/scripts/test_retrieval_s3.py`

**Interfaces:**
- Consumes: any canonical `frames.parquet` whose rows validate as `FrameRecord`.
- Produces: the existing visual embedding bundle and `DenseIndex` bundle; mapping still contains `frame_id`, `video_id`, `frame_idx`, `timestamp_ms`, `embedding_index`; `keyframe_order` is optional metadata.
- No `CustomEmbeddingBuilder` or custom FAISS builder is created.

- [ ] **Step 1: Add failing embedding test with a custom canonical row**

```python
def test_visual_builder_accepts_nullable_keyframe_order(tmp_path):
    frames = pd.DataFrame([{
        "frame_id": "V1_raw1fps_000000025",
        "video_id": "V1",
        "frame_idx": 25,
        "keyframe_order": None,
        "timestamp_ms": 1000,
        "fps": 25.0,
        "image_path": "custom_1fps/V1/25.jpg",
        "width": 64,
        "height": 64,
    }])
    image_root = tmp_path / "custom_1fps" / "V1"
    image_root.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(image_root / "25.jpg")
    frame_table = tmp_path / "frames.parquet"
    frames.to_parquet(frame_table, index=False)
    builder = EmbeddingArtifactBuilder(
        frames_path=frame_table,
        dataset_root=tmp_path,
        output_dir=tmp_path / "out",
        encoder_config=EncoderConfig(batch_size=1),
        encoder=CountingEncoder(),
        strict=True,
        resume=False,
        shard_size=1,
    )
    builder.run()
    mapping = pd.read_parquet(builder.mapping_file)
    assert pd.isna(mapping.loc[0, "keyframe_order"])
```

- [ ] **Step 2: Run the focused test and verify the current `int(None)`/required-identity failure**

```bash
PYTHONPATH=src python -m pytest \
  tests/retrieval/test_visual_embedding_resume.py::test_visual_builder_accepts_nullable_keyframe_order -v
```

Expected: FAIL because `_REQUIRED_FRAME_COLUMNS` and `_append_vectors()` require non-null `keyframe_order`.

- [ ] **Step 3: Make visual embedding identity generic**

Change required identity columns to:

```python
_REQUIRED_FRAME_COLUMNS = (
    "frame_id",
    "video_id",
    "frame_idx",
    "timestamp_ms",
    "image_path",
)
```

Validate `keyframe_order` only when present. In mapping append:

```python
keyframe_order = record.get("keyframe_order")
row = {
    "frame_id": record["frame_id"],
    "video_id": record["video_id"],
    "frame_idx": int(record["frame_idx"]),
    "embedding_index": position,
    "timestamp_ms": int(record["timestamp_ms"]),
}
if keyframe_order is not None and not pd.isna(keyframe_order):
    row["keyframe_order"] = int(keyframe_order)
else:
    row["keyframe_order"] = None
```

Keep exact row/vector alignment and strict unreadable-image failure semantics unchanged.

- [ ] **Step 4: Split generic index preflight from BTC-only organizer preflight**

In `scripts/build_retrieval_indexes.py`, generic visual build must validate:

```text
unique frame_id
non-null video_id/frame_idx/timestamp_ms/image_path
asset readability
embedding count == mapping count
```

Only BTC-specific preparation paths may additionally require:

```text
keyframe_order
BTC mapping equality
expected BTC frame count
```

Do not run BTC mapping checks when the configured frames artifact is custom/combined.

- [ ] **Step 5: Run visual/index regression tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/retrieval/test_visual_embedding_resume.py \
  tests/unit/retriever/test_index_artifact_integrity.py \
  tests/test_embedding_pipeline.py \
  tests/test_faiss_index.py \
  tests/scripts/test_retrieval_s3.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/retrieval/embedding/artifacts.py \
        scripts/build_retrieval_indexes.py \
        tests/retrieval/test_visual_embedding_resume.py \
        tests/scripts/test_retrieval_s3.py
git commit -m "refactor: make visual index builders frame-store generic"
```

---

### Task 5: Add Same-Channel BTC/Custom Visual Corpus Fusion

**Files:**
- Create: `src/hcmai/retrieval/retriever/dense/corpus.py`
- Modify: `src/hcmai/retrieval/retriever/dense/__init__.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Create: `tests/retrieval/test_visual_corpus_retriever.py`

**Interfaces:**
- Consumes: one shared visual query encoder and mapping `{corpus_name: DenseIndex}`.
- Produces: one retriever implementing the existing `Retriever` batch/search interface and emitting ordinary `RetrievalCandidate` rows with `RetrievalSource.VISUAL`.
- Same-channel merge score is deterministic reciprocal-rank fusion across visual corpora. Raw per-corpus rank/score lives in candidate metadata; `source_scores[VISUAL]` contains the merged visual-channel score consumed downstream.

- [ ] **Step 1: Write failing tests proving one encoding and two index searches**

```python
def test_visual_corpus_retriever_encodes_query_once_and_searches_both_indexes():
    retriever = VisualCorpusRetriever(
        encoder=fake_encoder,
        indexes={"btc": btc_index, "custom": custom_index},
        rrf_k=60,
    )
    result = retriever.search("red car", top_k=10)
    assert fake_encoder.calls == 1
    assert btc_index.search_calls == 1
    assert custom_index.search_calls == 1
    assert all(RetrievalSource.VISUAL in row.source_scores for row in result)


def test_visual_corpus_retriever_preserves_corpus_provenance(
    fake_encoder, btc_index, custom_index
):
    retriever = VisualCorpusRetriever(
        encoder=fake_encoder,
        indexes={"btc": btc_index, "custom": custom_index},
        rrf_k=60,
    )
    result = retriever.search("red car", top_k=10)
    assert result.candidates[0].metadata["visual_corpora"] in (
        ["btc"], ["custom"], ["btc", "custom"]
    )
```

Also test exact `frame_id` dedup when the same ID appears accidentally in both indexes; the higher/fused candidate must remain one row and retain both provenance entries.

- [ ] **Step 2: Run tests and confirm class is missing**

```bash
PYTHONPATH=src python -m pytest tests/retrieval/test_visual_corpus_retriever.py -v
```

Expected: FAIL import error.

- [ ] **Step 3: Implement `VisualCorpusRetriever` without changing `RetrievalSource`**

Core shape:

```python
class VisualCorpusRetriever:
    source = RetrievalSource.VISUAL
    source_family: SourceFamily = "visual"

    def __init__(
        self,
        encoder: TextEmbeddingAdapter,
        indexes: Mapping[str, DenseIndex],
        *,
        rrf_k: int,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
    ) -> None:
        if not indexes:
            raise ValueError("indexes must not be empty")
        self.encoder = encoder
        self.indexes = dict(indexes)
        self.rrf_k = rrf_k
        self.embedding_cache = embedding_cache
        self.prompt_version = prompt_version
        _validate_compatible_visual_indexes(self.indexes, encoder)

    def encode(self, query_texts: list[str]) -> QueryEmbeddingBatch:
        return encode_query_batch(
            query_texts, self.encoder, "visual", self.embedding_cache, self.prompt_version
        )

    def search_vectors(
        self,
        query_batch: QueryEmbeddingBatch,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        del query_type
        per_corpus = {
            name: _search_one_visual_index(index, query_batch, top_k, filters)
            for name, index in self.indexes.items()
        }
        return _merge_visual_results(per_corpus, top_k=top_k, rrf_k=self.rrf_k)
```

Do not pass two VISUAL jobs to `RRFFusionRetriever`; this class collapses them to one VISUAL result before multimodal fusion.

- [ ] **Step 4: Add a `RetrievalService.from_fast_track_indexes` parameter for multiple visual corpora**

Replace the single-index input with an interface that remains backwards compatible at the call site:

```python
visual_indexes: Mapping[str, DenseIndex]
```

If the mapping has one index, use `DenseRetriever`; if it has two, use `VisualCorpusRetriever`. Context/ASR retrievers and `RRFFusionRetriever` remain unchanged.

- [ ] **Step 5: Run visual-corpus plus fusion regressions**

```bash
PYTHONPATH=src python -m pytest \
  tests/retrieval/test_visual_corpus_retriever.py \
  tests/retrieval/test_fast_track_retrieval_composition.py \
  tests/unit/retriever/test_dense_index_score_subset.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/retrieval/retriever/dense/corpus.py \
        src/hcmai/retrieval/retriever/dense/__init__.py \
        src/hcmai/retrieval/retriever/pipeline.py \
        tests/retrieval/test_visual_corpus_retriever.py \
        tests/retrieval/test_fast_track_retrieval_composition.py
git commit -m "feat: fuse btc and custom visual corpora"
```

---

### Task 6: Persist Soft Boundary Priors Without Adding an Event Schema

**Files:**
- Create: `src/hcmai/data/preprocessing/boundaries.py`
- Modify: `src/hcmai/data/preprocessing/models.py` only where needed to expose score arrays cleanly
- Modify: `src/hcmai/data/preprocessing/video.py` only where needed to reuse timing analysis
- Create: `scripts/build_boundary_priors.py`
- Modify: `scripts/script_args.py`
- Create: `tests/preprocessing/test_boundary_priors.py`

**Interfaces:**
- Consumes: custom 1-FPS `frames.parquet`, custom visual `frame_mapping.parquet` + vectors, raw video paths for TransNetV2/EfficientGEBD.
- Produces: `boundaries.parquet` with module-local columns:

```text
video_id: str
timestamp_ms: int
shot_score: float | NaN
gebd_score: float | NaN
visual_change: float | NaN
asr_change: float | NaN
boundary_strength: float
```

`NaN` means not evaluated; it must not be converted to `0.0` before aggregation.

- [ ] **Step 1: Write failing signal-composition tests**

```python
def test_boundary_strength_ignores_not_evaluated_signals():
    row = combine_boundary_signals(
        shot_score=0.8,
        gebd_score=float("nan"),
        visual_change=0.4,
        asr_change=float("nan"),
    )
    assert row == pytest.approx(0.8)


def test_visual_change_uses_normalized_cosine_distance():
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, 1.0], dtype=np.float32)
    assert visual_change_score(left, right) == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

```bash
PYTHONPATH=src python -m pytest tests/preprocessing/test_boundary_priors.py -v
```

- [ ] **Step 3: Implement signal composition and vector-change projection**

```python
def combine_boundary_signals(*scores: float) -> float:
    available = [score for score in scores if math.isfinite(score)]
    return max(available) if available else float("nan")


def visual_change_score(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right))  # vectors are already L2 normalized
    return max(0.0, min(1.0, (1.0 - cosine) / 2.0))
```

- [ ] **Step 4: Build per-video boundary rows at 1-FPS timestamps**

Use existing TransNetV2/EfficientGEBD model adapters to score raw-video sequences, then project each signal to the nearest custom 1-FPS `timestamp_ms`. Join consecutive visual-embedding changes by `frame_id`/`timestamp_ms`. Set `asr_change=NaN` in this Batch-2 builder unless an already available ASR semantic-change artifact is explicitly supplied; do not fabricate zero evidence.

- [ ] **Step 5: Add an atomic/resumable boundary CLI**

```bash
PYTHONPATH=src python scripts/build_boundary_priors.py \
  --frames data/custom_1fps/frames.parquet \
  --visual-index artifacts/indexes/custom_visual \
  --videos-root data/videos \
  --output artifacts/boundaries/custom_1fps/boundaries.parquet
```

The script must support a bounded `--limit` smoke run and must not overwrite a complete full-corpus artifact with a limited run.

- [ ] **Step 6: Run boundary unit tests**

```bash
PYTHONPATH=src python -m pytest tests/preprocessing/test_boundary_priors.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/data/preprocessing/boundaries.py \
        src/hcmai/data/preprocessing/models.py \
        src/hcmai/data/preprocessing/video.py \
        scripts/build_boundary_priors.py scripts/script_args.py \
        tests/preprocessing/test_boundary_priors.py
git commit -m "feat: build soft temporal boundary priors"
```

---

### Task 7: Make `SceneCandidate` Assembly Boundary-Aware but Soft

**Files:**
- Create: `src/hcmai/temporal/boundaries.py`
- Modify: `src/hcmai/temporal/aligners/scene.py`
- Modify: `src/hcmai/temporal/core.py`
- Modify: `src/hcmai/temporal/ports.py` only if constructor typing requires a protocol
- Modify: `src/hcmai/common/config.py:305-370`
- Modify: `tests/unit/temporal/test_query_evidence.py`
- Modify: `tests/integration/test_progressive_temporal_core.py`
- Create: `tests/unit/temporal/test_boundary_scene_alignment.py`

**Interfaces:**
- Consumes: existing `FrameEvidence`, existing `TemporalQueryPlan`, optional persisted boundary artifact.
- Produces: ordinary existing `SceneCandidate` objects.
- Runtime helper:

```python
class BoundaryPriorIndex:
    @classmethod
    def load(cls, path: Path) -> "BoundaryPriorIndex":
        table = pd.read_parquet(path)
        return cls(table)

    def max_between(
        self, video_id: str, left_ms: int, right_ms: int
    ) -> float | None:
        values = self._strengths_between(video_id, left_ms, right_ms)
        return max(values) if values else None
```

- [ ] **Step 1: Add config fields with bounded defaults**

```python
scene_boundary_enabled: bool = True
scene_boundary_threshold: float = Field(default=0.65, ge=0, le=1)
scene_neighbor_expand_gap_ms: int = Field(default=3_000, ge=0)
scene_neighbor_expand_span_ms: int = Field(default=30_000, gt=0)
```

Validation must ensure neighbor expansion span is not below `scene_max_span_ms` if expansion is intended to recover split scenes.

- [ ] **Step 2: Write failing tests for boundary splitting and soft recombination**

```python
def test_strong_boundary_splits_initial_cluster():
    evidence = [_evidence(0), _evidence(1000), _evidence(2000)]
    priors = FakeBoundaryPrior({("V", 1000, 2000): 0.9})
    scenes = cluster_video_evidence(
        "V",
        evidence,
        max_gap_ms=5_000,
        max_span_ms=20_000,
        boundary_prior=priors,
        boundary_threshold=0.65,
    )
    assert [(s.start_ms, s.end_ms) for s in scenes] == [(0, 1000), (2000, 2000)]


def test_adjacent_scenes_are_offered_as_combined_candidate():
    evidence = [_evidence(0), _evidence(1000), _evidence(2000)]
    priors = FakeBoundaryPrior({("V", 1000, 2000): 0.9})
    scenes = cluster_video_evidence(
        "V",
        evidence,
        max_gap_ms=5_000,
        max_span_ms=20_000,
        boundary_prior=priors,
        boundary_threshold=0.65,
        neighbor_expand_gap_ms=3_000,
        neighbor_expand_span_ms=30_000,
    )
    assert any(scene.start_ms == 0 and scene.end_ms == 2000 for scene in scenes)
```

- [ ] **Step 3: Run the tests and verify current gap/span-only clustering fails**

```bash
PYTHONPATH=src python -m pytest tests/unit/temporal/test_boundary_scene_alignment.py -v
```

- [ ] **Step 4: Implement `BoundaryPriorIndex`**

Load Parquet once at service startup, group by video, sort by `timestamp_ms`, and answer maximum finite `boundary_strength` in `(left_ms, right_ms]`. Return `None` when no boundary was evaluated in the interval.

- [ ] **Step 5: Extend `cluster_video_evidence` with optional priors**

New keyword-only arguments:

```python
boundary_prior: BoundaryPriorIndex | None = None
boundary_threshold: float = 1.0
```

Split when any of these holds:

```text
adjacent gap > scene_max_gap_ms
scene span > scene_max_span_ms
finite boundary strength >= threshold
```

Add reason label `boundary_split` only to scenes actually separated by a finite prior.

- [ ] **Step 6: Generate adjacent union candidates in `ProgressiveSceneAligner.align`**

After initial clustering, add unions of immediately adjacent scenes when:

```text
gap <= scene_neighbor_expand_gap_ms
combined span <= scene_neighbor_expand_span_ms
```

Deduplicate by `(video_id, start_ms, end_ms, frame_ids)` before scoring. The union is a candidate, not a forced merge; `score_scene()` decides whether it ranks.

- [ ] **Step 7: Inject the boundary index through the existing temporal core**

`TemporalEvidenceCore.__init__` accepts `boundary_prior: BoundaryPriorIndex | None = None` and passes it to `ProgressiveSceneAligner`. Do not add a second temporal facade.

- [ ] **Step 8: Run temporal regression suite**

```bash
PYTHONPATH=src python -m pytest \
  tests/unit/temporal/test_boundary_scene_alignment.py \
  tests/unit/temporal/test_query_evidence.py \
  tests/unit/temporal/test_scoring_relations.py \
  tests/integration/test_progressive_temporal_core.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/hcmai/temporal/boundaries.py \
        src/hcmai/temporal/aligners/scene.py \
        src/hcmai/temporal/core.py src/hcmai/temporal/ports.py \
        src/hcmai/common/config.py \
        tests/unit/temporal/test_boundary_scene_alignment.py \
        tests/unit/temporal/test_query_evidence.py \
        tests/integration/test_progressive_temporal_core.py
git commit -m "feat: add soft boundary-aware scene assembly"
```

---

### Task 8: Add Training-Free Late Interaction and Deterministic Frame Refinement

**Files:**
- Modify: `src/hcmai/common/config.py` progressive-scene section
- Modify: `src/hcmai/temporal/scoring.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py:345-389`
- Modify: `tests/unit/temporal/test_scoring_relations.py`
- Modify: `tests/unit/orchestration/test_kis_reranking.py`

**Interfaces:**
- Consumes: existing normalized per-frame `FrameEvidence.unit_scores`.
- Produces: existing `SceneCandidate.unit_scores/semantic_score/final_score` and one representative existing `RetrievalCandidate` per scene.
- Config:

```python
scene_semantic_pooling: Literal["max", "logmeanexp"] = "logmeanexp"
scene_pool_temperature: float = Field(default=0.10, gt=0, le=1)
```

- [ ] **Step 1: Write failing unit-pooling tests**

```python
def test_max_pool_matches_existing_behavior():
    assert pool_unit_scores([0.2, 0.9, 0.4], mode="max", temperature=0.1) == 0.9


def test_logmeanexp_rewards_multiple_supporting_frames_without_exceeding_max():
    pooled = pool_unit_scores([0.8, 0.79, 0.1], mode="logmeanexp", temperature=0.1)
    assert 0.79 < pooled <= 0.8
```

Implement stable log-mean-exp, not plain log-sum-exp, so adding duplicate frames cannot arbitrarily increase score above the best normalized similarity.

- [ ] **Step 2: Run the focused tests and confirm helper is missing**

```bash
PYTHONPATH=src python -m pytest tests/unit/temporal/test_scoring_relations.py -v
```

- [ ] **Step 3: Replace hard `max(values)` semantic aggregation with configurable pooling**

Current line-level behavior:

```python
matched_scores = [max(values) for values in scores_by_unit.values() if values]
```

becomes:

```python
unit_scores = {
    unit_id: pool_unit_scores(
        values,
        mode=config.scene_semantic_pooling,
        temperature=config.scene_pool_temperature,
    )
    for unit_id, values in scores_by_unit.items()
    if values
}
matched_scores = list(unit_scores.values())
```

Preserve existing UNKNOWN/evaluated/no-match coverage semantics.

- [ ] **Step 4: Refine representative-frame selection using all scene query-unit support**

Replace `item.score`-only selection with a deterministic support score:

```python
def _frame_support(item: FrameEvidence, scene: SceneCandidate) -> float:
    values = [item.unit_scores[u] for u in scene.unit_scores if u in item.unit_scores]
    return sum(values) / len(values) if values else item.score
```

Tie-break in this exact order:

```text
higher frame support
higher original item.score
closer to scene midpoint
lower frame_idx
frame_id lexical order
```

This keeps final identity deterministic while allowing a frame supporting several query units to outrank a single spike.

- [ ] **Step 5: Add metadata needed for paper/debug inspection without changing contracts**

Representative candidate metadata includes:

```text
scene_id
scene_scores
frame_ids
selected_frame_support
scene_pooling
```

Do not add these as Pydantic schema fields; they remain bounded `RetrievalCandidate.metadata` diagnostics.

- [ ] **Step 6: Run scoring/KIS tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/unit/temporal/test_scoring_relations.py \
  tests/unit/orchestration/test_kis_reranking.py \
  tests/integration/test_kis_golden_path.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/common/config.py src/hcmai/temporal/scoring.py \
        src/hcmai/orchestration/workflows/kis.py \
        tests/unit/temporal/test_scoring_relations.py \
        tests/unit/orchestration/test_kis_reranking.py
git commit -m "feat: add training-free scene late interaction"
```

---

### Task 9: Wire Runtime Ablation Modes Through Existing Setup

**Files:**
- Modify: `src/hcmai/common/config.py:202-245,305-384`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `configs/baseline.yaml`
- Create: `configs/batch2/btc.yaml`
- Create: `configs/batch2/custom.yaml`
- Create: `configs/batch2/fused.yaml`
- Create: `configs/batch2/fused-frame-only.yaml`
- Create: `configs/batch2/fused-no-boundary.yaml`
- Create: `configs/batch2/fused-max-pool.yaml`
- Modify: `tests/orchestration/test_fast_track_setup.py`
- Create: `tests/orchestration/test_batch2_ablation_config.py`

**Interfaces:**
- Consumes: primary/secondary visual index paths, frames path selected by each YAML, optional boundary artifact.
- Produces: one `SearchService` using the same `RetrievalService`/`TemporalEvidenceCore` classes for every ablation.

- [ ] **Step 1: Add minimal index/search config fields**

Extend `IndexConfig`:

```python
secondary_visual_path: Path | None = None
visual_corpus_mode: Literal["btc", "custom", "fused"] = "btc"
```

Extend `ProgressiveSearchConfig`:

```python
scene_enabled: bool = True
boundary_path: Path | None = None
```

Do not add source-specific enum/schema types.

- [ ] **Step 2: Write config-validation tests**

Required invariants:

```text
mode=btc/custom => secondary_visual_path may be absent
mode=fused => secondary_visual_path is required
scene_boundary_enabled=true => boundary_path must exist at runtime startup, otherwise warn and disable only if boundary is not configured as required
scene_enabled=false => KIS uses frame-only candidates from the same retrieval service
```

- [ ] **Step 3: Update `orchestration/setup.py` to load visual index mapping**

Build:

```python
visual_indexes = {
    settings.index.visual_corpus_mode if settings.index.visual_corpus_mode != "fused" else "btc":
        RetrievalService.load_index(
            index_dir,
            subset_search_threshold=settings.index.subset_search_threshold,
        )
}
if settings.index.visual_corpus_mode == "fused":
    assert secondary_path is not None
    visual_indexes["custom"] = RetrievalService.load_index(
        secondary_path,
        subset_search_threshold=settings.index.subset_search_threshold,
    )
```

Use one visual encoder configured against the primary index; explicitly validate all visual indexes have the same model name and embedding dimension before constructing `RetrievalService.from_fast_track_indexes(visual_indexes=visual_indexes, visual_encoder=visual_encoder, context_index=context, asr_segment_index=asr_segment, text_encoder=text_encoder, frame_store=frame_store, fusion=settings.search.fusion, cache_config=settings.retrieval_cache)`.

- [ ] **Step 4: Load boundary priors once at startup**

When `scene_boundary_enabled` and `boundary_path` are configured, load `BoundaryPriorIndex` and inject it into `TemporalEvidenceCore`. Startup failure is explicit when the active Batch-2 config marks boundary priors required; no online rebuild fallback is permitted.

- [ ] **Step 5: Implement frame-only ablation without a legacy pipeline**

In `KISPipeline.execute`, when `config.progressive.scene_enabled` is false, use the current `RetrievalService.search()` result and materialize/rerank it through the same KIS pipeline stages. The runtime no longer switches to a legacy localization architecture.

This is an ablation flag, not a second KIS implementation.

- [ ] **Step 6: Create six concrete Batch-2 YAMLs**

Each YAML points to the matching runtime FrameStore:

```text
btc.yaml                  -> BTC frames + BTC visual index
custom.yaml               -> custom frames + custom visual index
fused.yaml                -> combined frames + BTC/custom indexes
fused-frame-only.yaml     -> combined frames + fused visual, scene_enabled=false
fused-no-boundary.yaml    -> combined frames + fused visual, boundary off
fused-max-pool.yaml       -> combined frames + fused visual, max pooling
```

All other model/fusion/reranker settings are copied from the same baseline config so the ablation changes only its named variable.

- [ ] **Step 7: Run setup/config tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/orchestration/test_fast_track_setup.py \
  tests/orchestration/test_batch2_ablation_config.py \
  tests/integration/test_progressive_temporal_core.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/common/config.py src/hcmai/orchestration \
        configs/baseline.yaml configs/batch2 tests/orchestration
git commit -m "feat: wire batch2 retrieval ablation modes"
```

---

### Task 10: Upgrade the Existing Evaluation Harness for Batch-2/Postmortem/Paper Use

**Files:**
- Modify: `src/hcmai/retrieval/retriever/evaluation/benchmark.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py` metadata introspection
- Modify: `scripts/build_benchmark.py`
- Modify: `scripts/script_args.py`
- Create: `tests/retrieval/test_batch2_benchmark.py`
- Modify: `src/hcmai/common/observability/README.md` only for artifact description

**Interfaces:**
- Consumes: existing `EvaluationQuery` JSONL, configured `SearchService`/retrieval path, authoritative frame store for gold-frame timestamp/video lookup.
- Produces one output directory per run:

```text
config.yaml
metrics.json
per_query.csv
failures.csv
trace_summary.json
```

No new evaluation schema is required for Batch 2.

- [ ] **Step 1: Write failing benchmark tests for video recall and temporal distance**

```python
def test_benchmark_reports_correct_video_recall_and_first_temporal_distance(
    tmp_path: Path, frame_store: FrameStore, retrieval_service: RetrievalService
) -> None:
    query = EvaluationQuery(
        query_id="q1",
        query="red car",
        language=QueryLanguage.ENGLISH,
        task_type=TaskType.KIS,
        difficulty=QueryDifficulty.HARD,
        gold_frame_ids=["gold-frame"],
        gold_video_ids=["V1"],
    )
    benchmark = RetrievalBenchmark(
        retrieval_service,
        [query],
        tmp_path,
        frame_store=frame_store,
        run_name="fixture",
        top_k=10,
    )
    metrics = benchmark.run()
    assert "video_recall" in metrics
    assert "temporal_distance_ms" in pd.read_csv(
        tmp_path / "per_query.csv"
    ).columns
```

Gold timestamps are resolved from each `EvaluationQuery.gold_frame_ids` through a supplied gold `DataService`/FrameStore. For a candidate in a gold video, temporal distance is the minimum absolute difference from any gold-frame timestamp.

- [ ] **Step 2: Run test and confirm current benchmark lacks these fields**

```bash
PYTHONPATH=src python -m pytest tests/retrieval/test_batch2_benchmark.py -v
```

- [ ] **Step 3: Remove the single-index metadata assumption**

Current `RetrievalBenchmark.run()` calls `retriever.index_metadata`, which raises for fused retrieval. Add a read-only `RetrievalService.index_metadata_list` property that recursively returns metadata for every dense index. Benchmark aggregate index size is the sum of unique artifact sizes; report a list of dataset/model/index identities in `config.yaml`.

- [ ] **Step 4: Extend per-query metrics**

Record:

```text
first_gold_rank
first_gold_video_rank
video_hit@1/5/10/20/100
frame recall@1/5/10/20/100
temporal_distance_ms for first candidate in a gold video
retrieved_video_id at rank 1
retrieved_frame_id at rank 1
query_encoding_ms
candidate_retrieval_ms
fusion_ms
localization_ms
reranking_ms
total_ms
warning_count
```

Do not claim statistical significance from 20 queries; this harness is paired diagnostic evidence.

- [ ] **Step 5: Make `scripts/build_benchmark.py` load current app configuration**

Remove its obsolete lookup:

```python
config.get("models", {}).get("embedding", {})
```

Instead load `AppConfig`, `LLMServiceConfig`, and the same service setup used by production, then execute the configured KIS search path. This ensures BTC/custom/fused and scene ablations exercise production code rather than a separate dense-only harness.

- [ ] **Step 6: Add run identity to every output**

The benchmark config records:

```text
run_name
query fixture hash
frames parquet checksum/path
visual corpus mode
visual index metadata list
scene enabled
boundary enabled/path checksum
pooling mode + temperature
search/reranker config dump
```

- [ ] **Step 7: Run benchmark and observability tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/retrieval/test_batch2_benchmark.py \
  tests/orchestration/test_fast_track_setup.py \
  tests/unit/temporal/test_core_regressions.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/retrieval/retriever/evaluation/benchmark.py \
        src/hcmai/retrieval/retriever/pipeline.py \
        scripts/build_benchmark.py scripts/script_args.py \
        tests/retrieval/test_batch2_benchmark.py \
        src/hcmai/common/observability/README.md
git commit -m "feat: add batch2 paper-ready evaluation harness"
```

---

### Task 11: Batch-2 Integration Gate and Minimal Documentation Cleanup

**Files:**
- Modify: `src/hcmai/README.md`
- Modify: `src/hcmai/data/README.md`
- Modify: `src/hcmai/data/preprocessing/README.md`
- Modify: `src/hcmai/temporal/README.md`
- Modify: `scripts/README.md`
- Modify: `.gitignore` if present in the real repository; otherwise create it with generated/cache paths only
- Test: existing targeted suites listed below

**Interfaces:**
- Produces no new runtime interface. This task verifies the integrated architecture and makes repository docs describe one current path accurately.

- [ ] **Step 1: Run the complete targeted Batch-2 test set before documentation edits**

```bash
PYTHONPATH=src python -m pytest \
  tests/preprocessing \
  tests/data/test_btc_keyframe_map.py \
  tests/data/test_frame_store_merge.py \
  tests/retrieval/test_visual_embedding_resume.py \
  tests/retrieval/test_visual_corpus_retriever.py \
  tests/retrieval/test_fast_track_retrieval_composition.py \
  tests/retrieval/test_batch2_benchmark.py \
  tests/unit/temporal \
  tests/integration/test_progressive_temporal_core.py \
  tests/orchestration/test_fast_track_setup.py \
  tests/orchestration/test_batch2_ablation_config.py \
  tests/unit/orchestration/test_kis_reranking.py -v
```

Expected: all PASS.

- [ ] **Step 2: Run a three-video smoke artifact build**

Use the validated submission rule from Task 1:

```bash
PYTHONPATH=src python scripts/preprocess_videos.py \
  --config configs/preparation.s3.yaml \
  --limit 3
```

Then build the custom visual index with the existing unified builder and run one BTC, one custom, and one fused benchmark configuration against a tiny local query fixture. Verify:

```text
all retrieved frame IDs resolve in the active FrameStore
no online code opens raw videos
custom rows have keyframe_order=None
submission(frame_id) returns stored frame_idx
fused retrieval emits visual corpus provenance
scene result references only same-video FrameEvidence
```

- [ ] **Step 3: Update architecture documentation to match the implemented path**

The READMEs must show:

```text
BTC baseline + deterministic custom 1 FPS
same-channel visual corpus fusion
FrameRecord reuse
SceneCandidate as event-like runtime unit
boundary priors are soft
training-free late interaction
Batch-2 ablation configs
```

Remove documentation that presents adaptive DINO keyframe selection as the current competition preparation path.

- [ ] **Step 4: Keep generated junk out of repository navigation**

Ensure ignore rules cover at least:

```gitignore
**/__pycache__/
**/*.pyc
frontend/node_modules/
frontend/build/
.env
```

Do not delete source modules unrelated to Batch 2 in this task.

- [ ] **Step 5: Run the full repository test suite if dependencies are available**

```bash
PYTHONPATH=src python -m pytest -q
```

Expected: PASS. If optional model/runtime dependencies cause environment-only skips, record them explicitly in the Batch-2 release notes; genuine regressions are blockers.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/README.md src/hcmai/data/README.md \
        src/hcmai/data/preprocessing/README.md src/hcmai/temporal/README.md \
        scripts/README.md .gitignore
git commit -m "docs: document batch2 event-centered retrieval path"
```

---

## Full-Corpus Execution Order After Code Merge

These are artifact-generation operations, not additional implementation tasks. Run them only after Tasks 1–11 are green.

1. Validate BTC mapping and freeze the one exact submission-coordinate rule.
2. Build/resume the complete 1-FPS custom FrameStore.
3. Build/resume custom visual embeddings with the same SigLIP2 revision as the BTC visual index.
4. Build the custom FAISS visual index.
5. Build the combined BTC+custom frame metadata artifact.
6. Build boundary priors.
7. Launch service with `configs/batch2/btc.yaml`; run official-query benchmark.
8. Launch with `configs/batch2/custom.yaml`; run the same query fixture.
9. Launch with `configs/batch2/fused-frame-only.yaml`; run the same fixture.
10. Launch with `configs/batch2/fused-no-boundary.yaml`; run the same fixture.
11. Launch with `configs/batch2/fused-max-pool.yaml`; run the same fixture.
12. Launch with `configs/batch2/fused.yaml`; run the same fixture.
13. Compare per-query rows, not only aggregate means.
14. Ship Batch 2 only if identity/asset/submission gates pass and the selected configuration has no unacceptable paired regression versus BTC-only.

## Batch-3 Postmortem Outputs Required for the Paper

Use Batch-2 benchmark artifacts to produce these tables before making new architectural changes:

```text
Table A: BTC vs custom vs fused visual coverage
Table B: frame-only vs SceneCandidate localization
Table C: boundary off vs on
Table D: max vs log-mean-exp scene pooling
Table E: latency/index/storage cost by configuration
```

For every official query, assign one failure label from a fixed taxonomy:

```text
visual coverage miss
wrong video semantic match
right video / wrong temporal region
scene fragmentation
scene over-merge
final representative frame error
OCR/ASR evidence required
reranker inversion
submission/identity error
```

Batch-3/4 work is chosen from this error table; do not add a new subsystem merely because it is novel.

---

## Plan Self-Review

### Spec coverage

- 1-FPS deterministic custom timeline: Tasks 1–2.
- Existing `FrameRecord` reuse and no new frame schema: Tasks 2–4.
- Combined runtime frame authority: Task 3.
- Generic existing embedding/index builders: Task 4.
- BTC/custom/fused visual retrieval without a new modality: Task 5 + Task 9.
- Soft boundary signals: Task 6.
- Boundary-aware query-conditioned `SceneCandidate`: Task 7.
- Training-free late interaction and frame refinement: Task 8.
- Same-codepath ablations: Task 9.
- Paper-ready paired evaluation: Task 10.
- Explicit offline generation and release gate: Task 11 + full-corpus execution order.
- Minimal legacy cleanup only when directly replaced: Task 2 + Task 11.

### Type/interface consistency

- Custom frames are always `FrameRecord`.
- Runtime temporal localization returns `SceneCandidate`.
- Visual corpus fusion returns existing `RetrievalResult` / `RetrievalCandidate` with `RetrievalSource.VISUAL`.
- Boundary artifacts are module-local Parquet plus `BoundaryPriorIndex`, not a domain schema.
- Submission continues through `official_frame_idx(frame)` after offline `frame_idx` materialization.

### Scope control

This plan intentionally does **not** implement:

- persistent `TemporalEvent`/`EventStore`/event FAISS index;
- HCMAI-supervised attention training;
- 2-FPS or adaptive-burst extraction;
- full Caption/OCR/Object enrichment of all 1-FPS frames;
- Story/news-topic hierarchy;
- broad `FrameEnrichment` migration. The separate P2 cleanup has already removed
  the runtime `legacy_specialists` profile and VQA legacy-localization package;
  `FrameEnrichment` remains until its consumers are migrated.

Those are Batch-3/4 decisions only if Batch-2 evidence justifies them.
