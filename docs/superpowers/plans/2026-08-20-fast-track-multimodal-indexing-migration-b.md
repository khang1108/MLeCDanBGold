# HCMAI 2026 Fast-Track Multimodal Indexing Migration B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status (2026-08-23):** This document records the original migration plan.
> The later P2 cleanup removed the runtime rollback profile and
> `RetrievalService.from_indexes()`; remaining references to that rollback path
> are historical plan context, not supported startup behavior.

**Goal:** Rebuild the deleted Visual index, build FrameContext and segment-native ASR indexes from the artifacts that exist today, and make `Visual + FrameContext + projected ASR` the default online localization stack without rewriting the existing RRF, temporal core, or TRAKE alignment.

**Architecture:** Keep `DenseIndex` frame-native for Visual and FrameContext. Introduce a separate `SegmentDenseIndex` for ASR, then adapt ASR segment hits to canonical frames in `ASRSegmentRetriever` before they reach the existing frame-ID RRF. Run all heavy embedding work offline on the ThunderCompute RTX A6000 and publish only complete validated bundles. The online composition uses the fast-track path; legacy text-index builders are not a runtime rollback profile.

**Tech Stack:** Python 3.12+, Pydantic v2, pandas/Parquet, NumPy, FAISS `IndexFlatIP`, Hugging Face Transformers SigLIP2, SentenceTransformers BGE-M3, pytest, pyright, rsync/SSH for local↔ThunderCompute transfer.

**Spec:** `docs/superpowers/specs/2026-08-20-frame-context-asr-indexing-design.md`

## Global Constraints

- BTC keyframes are the canonical visual corpus; do not re-extract keyframes from video.
- BTC `map_keyframes/<video_id>.csv` is authoritative for `n`, `pts_time`, `fps`, and `frame_idx`.
- Preserve BTC `frame_idx` exactly; never recompute it from timestamp/FPS.
- Preserve BTC `fps` exactly; values such as `29.97` and `26.44` must not be snapped to 30/25.
- Derive runtime `timestamp_ms` deterministically from BTC `pts_time`; retain the mapping CSV as audit source.
- Duplicate `(video_id, frame_idx)` values are valid; internal joins/index identity use unique `frame_id`.
- For the inspected BTC mapping bundle, strict production Visual coverage is exactly 177,321 frames across 873 videos.
- `FrameContext` remains deterministic Caption + normalized OCR + Object summary; do not add ASR to it.
- ASR source-of-truth remains `TranscriptSegment`; the persisted ASR index identity is `segment_id`, never a fabricated `frame_id`.
- Default online sources after promotion are `VISUAL + CONTEXT + ASR`; Caption/OCR/frame-aligned-ASR are not online runtime sources.
- Do not modify `src/hcmai/temporal/**` or `src/hcmai/pipelines/trake/**` for this migration.
- Do not replace `RetrievalCandidate.frame_id` with a generic temporal identity.
- Do not introduce IVF/PQ or learned fusion; keep exact `IndexFlatIP` + existing RRF.
- Online serving must never regenerate indexes.
- Visual is required at startup. Context/ASR may be configured optional during stabilization.
- Use the pinned Visual model `google/siglip2-base-patch16-224` revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`.
- Use the pinned text model `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181` for both FrameContext and ASR/query embeddings.
- On the single A6000, build sequentially: SigLIP2 Visual first, release it, then load BGE-M3 once for Context followed by ASR.
- The source archive supplied for planning does not contain the real repository `tests/` tree or `.git`; execute this plan in the real working repository and merge the focused tests below into its existing test conventions.

---

## File Structure to Lock In

### Create

- `src/hcmai/data/ingestion/keyframe_map.py` — parse/validate BTC CSV mapping, join it to canonical frame rows, and build a portable keyframe-path projection for remote indexing.
- `src/hcmai/retrieval/retriever/artifacts.py` — checksum/fingerprint helpers and atomic directory publication.
- `src/hcmai/retrieval/retriever/segment/__init__.py` — segment retrieval exports.
- `src/hcmai/retrieval/retriever/segment/index.py` — persisted `SegmentDenseIndex` with interval filtering and per-video postings.
- `src/hcmai/retrieval/retriever/segment/projector.py` — deterministic ASR-segment→canonical-frame projection.
- `src/hcmai/retrieval/retriever/segment/retriever.py` — BGE query encoding, segment search, projection, frame-candidate materialization.
- `src/hcmai/retrieval/retriever/segment/artifacts.py` — ASR segment corpus embedding/index build.
- `scripts/build_retrieval_indexes.py` — one fast-track offline entry point for preflight/visual/context/asr/all/validate.
- `scripts/sync_thundercompute_indexes.sh` — parameterized rsync workflow; no machine-specific paths hard-coded.
- `configs/indexing.yaml` — dedicated offline build inputs/outputs/strictness.
- `configs/indexing.models.yaml` — pinned SigLIP2 + BGE-M3 encoder config usable by offline build and online startup.
- `tests/data/test_btc_keyframe_map.py`
- `tests/retrieval/test_index_artifact_integrity.py`
- `tests/retrieval/test_visual_embedding_resume.py`
- `tests/retrieval/test_context_index.py`
- `tests/retrieval/test_segment_dense_index.py`
- `tests/retrieval/test_segment_frame_projector.py`
- `tests/retrieval/test_asr_segment_retriever.py`
- `tests/retrieval/test_fast_track_retrieval_composition.py`
- `tests/orchestration/test_fast_track_setup.py`
- `docs/runbooks/thundercompute-index-build.md`

### Modify

- `src/hcmai/data/ingestion/btc.py` — replace FPS estimation/snapping with BTC mapping join.
- `src/hcmai/data/stores/frame.py` — expose deterministic per-video frame access for the projector.
- `src/hcmai/data/pipeline.py` — expose FrameContext iteration/text and transcript store access without legacy coercion.
- `src/hcmai/common/schemas/enum.py` — add `RetrievalSource.CONTEXT`.
- `src/hcmai/common/schemas/retrieval.py` — expose context score in response diagnostics only; keep `frame_id` invariant.
- `src/hcmai/common/config.py` — configure context/transcript paths, new index paths/profile, and five-source fusion weights.
- `src/hcmai/llm/config.py` — add generic `evidence_embedding` with backward-compatible fallback to `caption_embedding`.
- `src/hcmai/retrieval/embedding/artifacts.py` — strict complete Visual coverage + deterministic shard resume.
- `src/hcmai/retrieval/embedding/models/metadata.py` — add revision/source fingerprint fields needed for resume validation.
- `src/hcmai/retrieval/retriever/models/metadata.py` — add optional v2 provenance/checksums while reading old metadata.
- `src/hcmai/retrieval/retriever/dense/index.py` — write/verify optional v2 integrity fields; do not change frame semantics.
- `src/hcmai/retrieval/retriever/text/retriever.py` — add FrameContext corpus/retriever support.
- `src/hcmai/retrieval/retriever/text/artifacts.py` — build Context from `FrameContextStore` with BGE-M3.
- `src/hcmai/retrieval/retriever/pipeline.py` — compose Visual + Context + ASR-segment retrievers while preserving legacy factory.
- `src/hcmai/orchestration/setup.py` — load new stores/indexes and select modern vs legacy profile.
- `scripts/build_embeddings.py` — route strict/resume options through the existing Visual compatibility entry point.
- `scripts/build_caption_index.py` — keep legacy CLI but allow `context` as a text source; ASR segment build goes through the new CLI.
- `configs/baseline.yaml` — default new paths/profile; legacy artifacts remain configured for rollback.
- `scripts/README.md`

### Do Not Modify

- `src/hcmai/temporal/**`
- `src/hcmai/pipelines/trake/**`
- `src/hcmai/retrieval/reranking/**`

---

### Task 1: Make BTC `map_keyframes` Authoritative Without Rewriting Internal Identity

**Files:**
- Create: `src/hcmai/data/ingestion/keyframe_map.py`
- Modify: `src/hcmai/data/ingestion/btc.py`
- Modify: `scripts/ingest_btc_keyframes.py`
- Test: `tests/data/test_btc_keyframe_map.py`

**Interfaces:**
- Consumes: existing source metadata rows containing `frame_id`, `video_id`, `keyframe_order`, `image_path`, `width`, `height`; BTC mapping directory containing `<video_id>.csv` with `n,pts_time,fps,frame_idx`.
- Produces: `load_btc_keyframe_map(mapping_root: Path) -> pd.DataFrame`; `join_btc_mapping(source_frames: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame`; `project_keyframe_paths(frames: pd.DataFrame, keyframes_root: Path) -> pd.DataFrame`; canonical rows with exact BTC `frame_idx`, `fps`, and derived `timestamp_ms` while preserving existing `frame_id`.

- [ ] **Step 1: Write failing mapping tests with the non-standard FPS and duplicate submission-coordinate cases**

```python
from pathlib import Path

import pandas as pd

from hcmai.data.ingestion.keyframe_map import load_btc_keyframe_map, join_btc_mapping


def test_mapping_preserves_exact_btc_coordinates(tmp_path: Path):
    root = tmp_path / "map_keyframes"
    root.mkdir()
    pd.DataFrame([
        {"n": 1, "pts_time": 1.001, "fps": 29.97, "frame_idx": 30},
        {"n": 2, "pts_time": 2.002, "fps": 29.97, "frame_idx": 60},
    ]).to_csv(root / "L01_V001.csv", index=False)

    source = pd.DataFrame([
        {"frame_id": "f1", "video_id": "L01_V001", "keyframe_order": 1},
        {"frame_id": "f2", "video_id": "L01_V001", "keyframe_order": 2},
    ])

    joined = join_btc_mapping(source, load_btc_keyframe_map(root))
    assert joined["fps"].tolist() == [29.97, 29.97]
    assert joined["frame_idx"].tolist() == [30, 60]
    assert joined["timestamp_ms"].tolist() == [1001, 2002]


def test_duplicate_submission_coordinates_are_allowed(tmp_path: Path):
    root = tmp_path / "map_keyframes"
    root.mkdir()
    pd.DataFrame([
        {"n": 1, "pts_time": 0.040, "fps": 25.0, "frame_idx": 1},
        {"n": 2, "pts_time": 0.079, "fps": 25.0, "frame_idx": 1},
    ]).to_csv(root / "L01_V001.csv", index=False)
    source = pd.DataFrame([
        {"frame_id": "f1", "video_id": "L01_V001", "keyframe_order": 1},
        {"frame_id": "f2", "video_id": "L01_V001", "keyframe_order": 2},
    ])
    joined = join_btc_mapping(source, load_btc_keyframe_map(root))
    assert joined["frame_id"].tolist() == ["f1", "f2"]
    assert joined["frame_idx"].tolist() == [1, 1]
```

Also add focused tests rejecting non-contiguous `n`, changing FPS inside one CSV, negative `pts_time`, and a missing `(video_id, keyframe_order)` mapping row. Add a portable-path test with two sorted keyframe image files under `keyframes/<video_id>/` and assert the returned projection rewrites only `image_path`, not `frame_id/frame_idx/timestamp_ms`.

- [ ] **Step 2: Run the focused test and verify it fails because the mapping module does not exist**

Run:

```bash
PYTHONPATH=.:src python -m pytest tests/data/test_btc_keyframe_map.py -q
```

Expected: import failure for `hcmai.data.ingestion.keyframe_map`.

- [ ] **Step 3: Implement strict mapping load and join**

Create `keyframe_map.py` with this public boundary:

```python
def load_btc_keyframe_map(mapping_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(mapping_root.glob("*.csv")):
        table = pd.read_csv(path)
        if list(table.columns) != ["n", "pts_time", "fps", "frame_idx"]:
            raise ValueError(f"Unexpected BTC mapping schema: {path}")
        expected = list(range(1, len(table) + 1))
        if table["n"].astype(int).tolist() != expected:
            raise ValueError(f"BTC mapping n must be contiguous 1..N: {path}")
        if (table["pts_time"] < 0).any() or (table["frame_idx"] < 0).any():
            raise ValueError(f"BTC mapping coordinates must be non-negative: {path}")
        fps_values = table["fps"].astype(float).unique()
        if len(fps_values) != 1 or float(fps_values[0]) <= 0:
            raise ValueError(f"BTC mapping fps must be one positive value per video: {path}")
        table = table.assign(video_id=path.stem, keyframe_order=table["n"].astype(int))
        rows.append(table)
    if not rows:
        raise ValueError(f"No BTC mapping CSV files found under {mapping_root}")
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["video_id", "keyframe_order"]).any():
        raise ValueError("BTC mapping contains duplicate video/keyframe order")
    return result


def join_btc_mapping(source_frames: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    joined = source_frames.merge(
        mapping[["video_id", "keyframe_order", "pts_time", "fps", "frame_idx"]],
        on=["video_id", "keyframe_order"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_btc"),
    )
    if joined["pts_time"].isna().any():
        raise ValueError("Canonical source contains frames missing from BTC mapping")
    joined["frame_idx"] = joined.pop("frame_idx_btc").astype("int64")
    joined["fps"] = joined["fps"].astype(float)
    joined["timestamp_ms"] = (joined["pts_time"].astype(float) * 1000.0).round().astype("int64")
    return joined
```

If the source metadata already has `frame_idx`, drop/rename it before merge so `frame_idx_btc` is unambiguous. Do not deduplicate on `(video_id, frame_idx)`.

Also implement a non-mutating remote path projection. For each video, list image files with suffix in `{.jpg,.jpeg,.png,.webp}` sorted by filename, require the image count to equal the frame count sorted by `keyframe_order`, and return a copied DataFrame whose `image_path` points to those staged files. This is the only supported way to bypass local absolute `image_path` values on ThunderCompute; do not rewrite the canonical Parquet just to change machines.

- [ ] **Step 4: Modify `BTCIngestionConfig` and `import_btc_frame_store()` to require/use the mapping root**

Add:

```python
mapping_root: Path
```

Update `scripts/ingest_btc_keyframes.py` with `--mapping-root` defaulting to `data/map_keyframes` and pass it to `BTCIngestionConfig`.

Delete `_compute_fps_per_video()`. Before `_build_canonical_rows`, call `join_btc_mapping(...)`. In `_build_canonical_rows`, use `row["fps"]` directly and preserve `row["frame_idx"]` exactly. Add BTC mapping counts and collision diagnostics to `manifest.json`.

- [ ] **Step 5: Run focused ingestion tests**

Run:

```bash
PYTHONPATH=.:src python -m pytest tests/data/test_btc_keyframe_map.py -q
pyright src/hcmai/data/ingestion
```

Expected: PASS; no standard-FPS snapping remains in the BTC path.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/data/ingestion scripts/ingest_btc_keyframes.py tests/data/test_btc_keyframe_map.py
git commit -m "fix(data): use btc media mapping as canonical coordinates"
```

---

### Task 2: Add Backward-Compatible Index Provenance, Checksums, and Atomic Publication

**Files:**
- Create: `src/hcmai/retrieval/retriever/artifacts.py`
- Modify: `src/hcmai/retrieval/retriever/models/metadata.py`
- Modify: `src/hcmai/retrieval/retriever/dense/index.py`
- Test: `tests/retrieval/test_index_artifact_integrity.py`

**Interfaces:**
- Consumes: complete files emitted by `DenseIndex.save()` / later `SegmentDenseIndex.save()`.
- Produces: optional v2 fields on `IndexMetadata`; `sha256_file(path) -> str`; `fingerprint_files(paths) -> str`; `publish_directory(staged: Path, destination: Path) -> Path`.

- [ ] **Step 1: Write failing round-trip and tamper tests**

```python
def test_v2_dense_index_rejects_tampered_vectors(tmp_path, tiny_dense_index):
    output = tiny_dense_index.save(tmp_path / "index")
    vectors = output / "vectors.npy"
    with vectors.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(IndexArtifactError, match="checksum"):
        DenseIndex.load(output)


def test_old_metadata_without_v2_fields_remains_readable(old_style_index_dir):
    loaded = DenseIndex.load(old_style_index_dir)
    assert loaded.metadata.schema_version == "dense-index-v1"
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_index_artifact_integrity.py -q
```

Expected: v2 metadata/checksum behavior is missing.

- [ ] **Step 3: Extend `IndexMetadata` with defaulted v2 provenance**

Use defaulted fields so old JSON remains readable:

```python
@dataclass
class IndexMetadata:
    dataset_version: str
    model_name: str
    index_type: str
    metric: str
    normalization: str
    embedding_dim: int
    vector_count: int
    build_time_sec: float
    index_size_bytes: int
    generated_at: str
    schema_version: str = "dense-index-v1"
    entity_kind: str = "frame"
    retrieval_source: str | None = None
    model_revision: str | None = None
    source_fingerprint: str | None = None
    config_fingerprint: str | None = None
    checksums: dict[str, str] | None = None
```

`from_dict()` must ignore no known fields and accept old dictionaries through the defaults.

- [ ] **Step 4: Implement file checksums and atomic directory publication**

```python
def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def publish_directory(staged: Path, destination: Path) -> Path:
    staged = staged.resolve()
    destination = destination.resolve()
    if not staged.is_dir():
        raise FileNotFoundError(staged)
    backup = destination.with_name(destination.name + ".backup")
    if backup.exists():
        raise RuntimeError(f"Stale publication backup exists: {backup}")
    try:
        if destination.exists():
            destination.replace(backup)
        staged.replace(destination)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if backup.exists():
            backup.replace(destination)
        raise
    return destination
```

Use a sibling staging directory so the final rename is on one filesystem.

- [ ] **Step 5: Make `DenseIndex.save()` write metadata last with checksums, and `load()` verify them when present**

Checksum at least:

```text
dense.index
frame_mapping.parquet
vectors.npy
posting_video_ids.json
posting_offsets.npy
posting_positions.npy
timestamps.npy
```

Set `schema_version="dense-index-v2"` for newly built indexes. `DenseIndex.load()` must keep reading v1 artifacts with `checksums=None` for rollback compatibility.

- [ ] **Step 6: Run tests and type checking**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_index_artifact_integrity.py -q
pyright src/hcmai/retrieval/retriever/models/metadata.py src/hcmai/retrieval/retriever/dense/index.py src/hcmai/retrieval/retriever/artifacts.py
```

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/retrieval/retriever tests/retrieval/test_index_artifact_integrity.py
git commit -m "feat(retrieval): add index integrity metadata"
```

---

### Task 3: Make the Visual SigLIP2 Build Strict and Resumable on the A6000

**Files:**
- Modify: `src/hcmai/retrieval/embedding/artifacts.py`
- Modify: `src/hcmai/retrieval/embedding/models/metadata.py`
- Modify: `scripts/build_embeddings.py`
- Test: `tests/retrieval/test_visual_embedding_resume.py`

**Interfaces:**
- Consumes: canonical `frames.parquet`, portable `dataset_root`/keyframe root, pinned `EncoderConfig`.
- Produces: deterministic visual embedding shards, compact `visual_embeddings.npy`, `frame_mapping.parquet`, metadata, then existing frame-native `DenseIndex`; strict mode refuses publication unless all expected frames are encoded exactly once.

- [ ] **Step 1: Write failing tests for strict coverage and shard resume**

Use an injected fake encoder so tests never load SigLIP2:

```python
def test_strict_visual_build_refuses_missing_image(tmp_path, fake_encoder, frame_table):
    builder = EmbeddingArtifactBuilder(
        frames_path=frame_table,
        dataset_root=tmp_path / "keyframes",
        output_dir=tmp_path / "out",
        encoder_config=EncoderConfig(batch_size=2),
        encoder=fake_encoder,
        strict=True,
    )
    with pytest.raises(RuntimeError, match="complete visual coverage"):
        builder.run()


def test_visual_build_reuses_valid_completed_shard(tmp_path, counting_encoder, frame_table):
    first = make_builder(tmp_path, counting_encoder, frame_table, shard_size=2)
    first.run()
    calls_after_first = counting_encoder.image_count
    second = make_builder(tmp_path, counting_encoder, frame_table, shard_size=2)
    second.run()
    assert counting_encoder.image_count == calls_after_first
```

Also test a shard with mismatched frame IDs is ignored and regenerated rather than appended.

- [ ] **Step 2: Run tests and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_visual_embedding_resume.py -q
```

- [ ] **Step 3: Extend `EmbeddingMetadata` with revision and source fingerprint defaults**

Add defaulted fields:

```python
model_revision: str | None = None
source_fingerprint: str | None = None
schema_version: str = "visual-embedding-v2"
```

Existing `from_dict()` remains backward-compatible.

- [ ] **Step 4: Add deterministic shard checkpoints**

Use fixed row ranges from canonical frame order. A shard contains the expected frame IDs and vectors:

```python
np.savez_compressed(
    shard_path,
    frame_ids=np.asarray(expected_ids, dtype=str),
    vectors=np.asarray(vectors, dtype=np.float32),
)
```

Write through `atomic_write`. On resume, reuse only when the saved `frame_ids` exactly equal the expected slice and vector dimension/count are valid. Never key resume by filename guesses.

- [ ] **Step 5: Compact shards in canonical order and enforce strict production coverage**

Before writing final files, require:

```python
if strict and (
    failed_frames
    or len(frame_mapping) != total_frames
    or len(set(row["frame_id"] for row in frame_mapping)) != total_frames
):
    raise RuntimeError("Visual build does not have complete visual coverage")
```

Include `keyframe_order` in the persisted frame mapping. The current mapping already contains `frame_id/video_id/frame_idx/timestamp_ms`; do not remove them.

- [ ] **Step 6: Route strict/resume settings through `build_embeddings.py`**

Add CLI flags through `script_args.py` if that is where shared arguments live:

```text
--strict / --no-strict       default strict
--resume / --no-resume       default resume
--shard-size <int>           default 2048
--dataset-root <path>        portable ThunderCompute keyframe root
```

`build_embeddings.py` must build into a sibling staging directory, load the completed `DenseIndex` once for validation, and call `publish_directory(...)` only after strict coverage and index integrity pass.

- [ ] **Step 7: Run tests and type checking**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_visual_embedding_resume.py tests/retrieval/test_index_artifact_integrity.py -q
pyright src/hcmai/retrieval/embedding scripts/build_embeddings.py
```

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/retrieval/embedding scripts/build_embeddings.py scripts/script_args.py tests/retrieval/test_visual_embedding_resume.py
git commit -m "feat(retrieval): make visual index build strict and resumable"
```

---

### Task 4: Introduce `CONTEXT` Source and Generic Evidence Encoder Configuration

**Files:**
- Modify: `src/hcmai/common/schemas/enum.py`
- Modify: `src/hcmai/common/schemas/retrieval.py`
- Modify: `src/hcmai/common/config.py`
- Modify: `src/hcmai/llm/config.py`
- Create: `configs/indexing.models.yaml`
- Create: `configs/indexing.yaml`
- Test: `tests/retrieval/test_context_index.py` (configuration tests in this file)

**Interfaces:**
- Consumes: existing legacy `caption_embedding` configuration.
- Produces: `RetrievalSource.CONTEXT`; `LLMServiceConfig.resolved_evidence_embedding`; `IndexConfig.context_path`; `IndexConfig.asr_segment_path`; `IndexConfig.profile` with `context_asr_segment|legacy_specialists`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_evidence_embedding_falls_back_to_caption_embedding():
    config = LLMServiceConfig(
        caption_embedding=EncoderConfig(backend="bge_m3", model_name="BAAI/bge-m3")
    )
    assert config.resolved_evidence_embedding.model_name == "BAAI/bge-m3"


def test_fusion_accepts_context_as_a_source():
    config = FusionConfig()
    assert RetrievalSource.CONTEXT in config.task_weights[TaskType.KIS]
```

Also parse a YAML with explicit `evidence_embedding` and assert it wins over `caption_embedding`.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_context_index.py -q
```

- [ ] **Step 3: Add the new retrieval source without removing legacy values**

```python
class RetrievalSource(str, Enum):
    VISUAL = "visual"
    CONTEXT = "context"
    CAPTION = "caption"
    OCR = "ocr"
    ASR = "asr"
```

Add `context: float | None = None` to `SearchScores`. Keep `RetrievalCandidate.frame_id` unchanged.

- [ ] **Step 4: Extend application config**

Add to `EnrichmentArtifactsConfig`:

```python
context_path: Path | None = Path("artifacts/enrichment/context/frame_context_v1.parquet")
transcripts_path: Path | None = Path("artifacts/enrichment/transcripts")
```

Add to `IndexConfig`:

```python
profile: Literal["context_asr_segment", "legacy_specialists"] = "context_asr_segment"
context_path: Path = Path("artifacts/indexes/context")
asr_segment_path: Path = Path("artifacts/indexes/asr_segments")
context_embedding_filename: str = "context_embeddings.npy"
asr_segment_embedding_filename: str = "asr_embeddings.npy"
asr_projection_max_gap_ms: int = Field(default=5000, ge=0)
```

Keep the old `caption_path/ocr_path/asr_path` fields for rollback. Expand fusion task-weight validation to all five enum values.

- [ ] **Step 5: Add generic evidence encoder config with compatibility fallback**

In `LLMServiceConfig`:

```python
evidence_embedding: EncoderConfig | None = None

@property
def resolved_evidence_embedding(self) -> EncoderConfig:
    return self.evidence_embedding or self.caption_embedding
```

Update `from_yaml()` to parse `visual_embedding`, `caption_embedding`, and `evidence_embedding` through `EncoderConfig.from_dict()` when present.

- [ ] **Step 6: Create the pinned model file**

`configs/indexing.models.yaml`:

```yaml
visual_embedding:
  backend: siglip
  model_name: google/siglip2-base-patch16-224
  revision: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
  device: cuda
  batch_size: 64
  image_size: 224
  dtype: float32

evidence_embedding:
  backend: bge_m3
  model_name: BAAI/bge-m3
  revision: 5617a9f61b028005a4858fdac845db406aefb181
  device: cuda
  batch_size: 64
  max_length: 8192
  dtype: float32

# Rollback compatibility: legacy text builders still read this field.
caption_embedding:
  backend: bge_m3
  model_name: BAAI/bge-m3
  revision: 5617a9f61b028005a4858fdac845db406aefb181
  device: cuda
  batch_size: 64
  max_length: 8192
  dtype: float32
```

Batch sizes are conservative starting values, not an A6000 maximum; the runbook may tune them after measuring VRAM.

- [ ] **Step 7: Create dedicated offline indexing config**

`configs/indexing.yaml`:

```yaml
dataset:
  version: hcmai2026_v1
  frames_path: artifacts/frame_store/frames.parquet
  frame_manifest: artifacts/frame_store/manifest.json
  keyframes_root: data/keyframes
  map_keyframes_root: data/map_keyframes
  context_path: artifacts/enrichment/context/frame_context_v1.parquet
  transcripts_path: artifacts/enrichment/transcripts
  expected_video_count: 873
  expected_frame_count: 177321

build:
  strict_visual_coverage: true
  resume: true
  visual_shard_size: 2048

indexes:
  visual: artifacts/indexes/visual
  context: artifacts/indexes/context
  asr_segments: artifacts/indexes/asr_segments

projection:
  max_projection_gap_ms: 5000
```

Paths may be overridden by CLI on ThunderCompute; do not rewrite this file with server-specific absolute paths.

- [ ] **Step 8: Run tests/type check and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_context_index.py -q
pyright src/hcmai/common/config.py src/hcmai/common/schemas src/hcmai/llm/config.py

git add src/hcmai/common src/hcmai/llm/config.py configs/indexing.yaml configs/indexing.models.yaml tests/retrieval/test_context_index.py
git commit -m "feat(config): add context and segment retrieval configuration"
```

---

### Task 5: Build the FrameContext BGE-M3 Index as a Frame-Native `DenseIndex`

**Files:**
- Modify: `src/hcmai/data/pipeline.py`
- Modify: `src/hcmai/retrieval/retriever/text/retriever.py`
- Modify: `src/hcmai/retrieval/retriever/text/artifacts.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Modify: `scripts/build_caption_index.py`
- Test: `tests/retrieval/test_context_index.py`

**Interfaces:**
- Consumes: `FrameContextStore`, canonical `FrameStore`, `TextEmbeddingAdapter` configured as BGE-M3.
- Produces: `ContextRetriever`; `build_context_index(...) -> DenseIndex`; `RetrievalService.build_context_artifacts(...) -> DenseIndex`; `artifacts/indexes/context/*`.

- [ ] **Step 1: Add failing corpus tests**

```python
def test_context_corpus_embeds_only_non_empty_context(data_service_with_context):
    texts, mapping = _context_corpus(data_service_with_context)
    assert texts == ["[CAPTION]\nA red cable car."]
    assert mapping["frame_id"].tolist() == ["f1"]
    assert mapping["timestamp_ms"].tolist() == [1000]


def test_context_index_is_frame_native(fake_bge, data_service_with_context, tmp_path):
    index = build_context_index(
        data_service_with_context,
        fake_bge,
        tmp_path,
        embeddings_filename="context_embeddings.npy",
        dataset_version="test-v1",
    )
    assert index.mapping["frame_id"].tolist() == ["f1"]
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_context_index.py -q
```

- [ ] **Step 3: Expose FrameContext iteration through `DataService`**

Add:

```python
def iter_frame_contexts(self) -> Iterator[FrameContext]:
    if self.context_store is None:
        return iter(())
    return self.context_store.iter_records()


def get_frame_context_text(self, frame_id: str) -> str | None:
    if self.context_store is None:
        return None
    try:
        return self.context_store.get_text(frame_id)
    except KeyError:
        return None
```

Do not add Context to `evidence_stores`; it remains its own typed store.

- [ ] **Step 4: Add `_context_corpus` and `ContextRetriever`**

```python
class ContextRetriever(DenseRetriever):
    def __init__(self, encoder, index, embedding_cache=None, prompt_version="query-v1"):
        super().__init__(
            encoder,
            index,
            RetrievalSource.CONTEXT,
            embedding_cache,
            prompt_version,
        )


def _context_corpus(data: DataService) -> tuple[list[str], pd.DataFrame]:
    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for context in data.iter_frame_contexts():
        text = data.get_frame_context_text(context.frame_id)
        if text is None:
            continue
        frame = data.get_frame(context.frame_id)
        rows.append({
            "frame_id": frame.frame_id,
            "video_id": frame.video_id,
            "frame_idx": frame.frame_idx,
            "keyframe_order": frame.keyframe_order,
            "timestamp_ms": frame.timestamp_ms,
            "embedding_index": len(texts),
        })
        texts.append(text)
    if not texts:
        raise ValueError("FrameContext artifact contains no usable context_text")
    return texts, pd.DataFrame(rows)
```

Use the existing `_normalized()` and `_encode_texts()` helpers.

- [ ] **Step 5: Add a dedicated Context artifact builder**

`build_context_artifacts(...)` must call:

```python
data = DataService.load(frames, context_path=context)
encoder_config = models.resolved_evidence_embedding
selected_encoder = encoder or EmbeddingService.create_text_adapter(encoder_config)
```

Then build/save a frame-native `DenseIndex` with `retrieval_source="context"`, `entity_kind="frame"`, and the source fingerprint from `frame_context_v1.parquet` + its manifest. Use `settings.index.context_embedding_filename`; build in a sibling staging directory, validate with `DenseIndex.load(staged)`, and atomically publish only after the round-trip succeeds.

- [ ] **Step 6: Export service/CLI compatibility**

Add `RetrievalService.build_context_artifacts(...)`. In `scripts/build_caption_index.py`, branch explicitly: `source == CONTEXT` calls `build_context_artifacts(...)`; legacy Caption/OCR/frame-ASR continue through `build_text_artifacts(...)`. Allow:

```bash
python scripts/build_caption_index.py --source context ...
```

for convenience, but do not make this CLI build segment-native ASR; that uses Task 7. When a remote embedding adapter is used for Context, pass remote source family `"text"`, not `"context"`, because the hosted embedding contract selects the BGE/caption-compatible encoder on `source == "text"`.

- [ ] **Step 7: Verify exact filtered search round-trip**

Add a test comparing `DenseIndex.search_filtered()` for one video/time window against direct `query @ vectors.T` over the same allowed positions.

- [ ] **Step 8: Run and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_context_index.py -q
pyright src/hcmai/data/pipeline.py src/hcmai/retrieval/retriever/text src/hcmai/retrieval/retriever/pipeline.py

git add src/hcmai/data/pipeline.py src/hcmai/retrieval/retriever/text src/hcmai/retrieval/retriever/pipeline.py scripts/build_caption_index.py tests/retrieval/test_context_index.py
git commit -m "feat(retrieval): build frame context dense index"
```

---

### Task 6: Introduce a Segment-Native Exact Dense Index for ASR

**Files:**
- Create: `src/hcmai/retrieval/retriever/segment/__init__.py`
- Create: `src/hcmai/retrieval/retriever/segment/index.py`
- Test: `tests/retrieval/test_segment_dense_index.py`

**Interfaces:**
- Consumes: normalized `float32` embeddings and mapping columns `embedding_index,segment_id,video_id,segment_index,start_ms,end_ms`.
- Produces: `SegmentDenseIndex.build/load/save/search/search_filtered/video_positions`; files `dense.index`, `segment_mapping.parquet`, `vectors.npy`, `posting_*`, `start_ms.npy`, `end_ms.npy`, `metadata.json`.

- [ ] **Step 1: Write failing identity/filter tests**

```python
def test_segment_index_rejects_duplicate_segment_ids():
    mapping = pd.DataFrame([
        {"embedding_index": 0, "segment_id": "s1", "video_id": "v1", "segment_index": 0, "start_ms": 0, "end_ms": 1000},
        {"embedding_index": 1, "segment_id": "s1", "video_id": "v1", "segment_index": 1, "start_ms": 1000, "end_ms": 2000},
    ])
    with pytest.raises(ValueError, match="duplicate segment_id"):
        SegmentDenseIndex.build(unit_vectors(2), mapping, dataset_version="v1", model_name="m")


def test_segment_filter_uses_half_open_overlap(tiny_segment_index):
    positions = tiny_segment_index.filtered_positions(
        SearchFilters(video_ids=["v1"], start_time_ms=1000, end_time_ms=2000)
    )
    assert tiny_segment_index.mapping.iloc[positions]["segment_id"].tolist() == ["s2"]
```

Fixture boundaries should include `[0,1000)`, `[1000,2000)`, and `[2000,3000)` so touching but non-overlapping segments are excluded.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_segment_dense_index.py -q
```

- [ ] **Step 3: Implement build-time mapping validation**

Require:

```python
required = {
    "embedding_index", "segment_id", "video_id", "segment_index", "start_ms", "end_ms"
}
if mapping["segment_id"].duplicated().any():
    raise ValueError("mapping contains duplicate segment_id values")
if (mapping["end_ms"] <= mapping["start_ms"]).any():
    raise ValueError("segment mapping requires positive duration")
```

Use exact `faiss.IndexFlatIP`; normalized vectors remain the contract.

- [ ] **Step 4: Persist vectors/postings/start/end arrays and v2 metadata**

Use the same integrity helpers as Task 2. New metadata values:

```text
schema_version = dense-index-v2
entity_kind = segment
retrieval_source = asr
```

- [ ] **Step 5: Implement interval-aware filtering**

```python
if filters.start_time_ms is not None:
    positions = positions[self.end_ms[positions] > filters.start_time_ms]
if filters.end_time_ms is not None:
    positions = positions[self.start_ms[positions] < filters.end_time_ms]
if (
    filters.start_time_ms is not None
    and filters.end_time_ms is not None
    and filters.start_time_ms == filters.end_time_ms
):
    return np.empty(0, dtype=np.int64)
```

Then reuse `exact_subset_search()` for restricted search.

- [ ] **Step 6: Add global-vs-subset exactness and round-trip tests**

Compare restricted results to brute-force matrix multiplication over the selected positions. Save/load and assert segment mapping has no `frame_id` column.

- [ ] **Step 7: Run and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_segment_dense_index.py tests/retrieval/test_index_artifact_integrity.py -q
pyright src/hcmai/retrieval/retriever/segment

git add src/hcmai/retrieval/retriever/segment tests/retrieval/test_segment_dense_index.py
git commit -m "feat(retrieval): add segment native dense index"
```

---

### Task 7: Build the ASR Segment BGE-M3 Corpus and Index

**Files:**
- Create: `src/hcmai/retrieval/retriever/segment/artifacts.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Test: `tests/retrieval/test_asr_segment_retriever.py` (artifact-build cases)

**Interfaces:**
- Consumes: `TranscriptStore`, `TextEmbeddingAdapter` using `resolved_evidence_embedding`.
- Produces: `build_asr_segment_artifacts(...) -> SegmentDenseIndex`; `RetrievalService.build_asr_segment_artifacts(...)`; `artifacts/indexes/asr_segments/*`.

- [ ] **Step 1: Write failing corpus-build tests**

```python
def test_asr_segment_corpus_preserves_timeline_identity(transcript_store):
    texts, mapping = build_segment_corpus(transcript_store)
    assert texts == ["hello world"]
    assert mapping.loc[0, "segment_id"] == "v1:0"
    assert mapping.loc[0, "start_ms"] == 1000
    assert mapping.loc[0, "end_ms"] == 2000
    assert "frame_id" not in mapping.columns
```

Include one `FAILED` segment fixture and assert it is excluded from embedding.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_asr_segment_retriever.py -q
```

- [ ] **Step 3: Implement the segment corpus builder**

```python
def build_segment_corpus(store: TranscriptStore) -> tuple[list[str], pd.DataFrame]:
    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for segment in store.iter_records():
        if segment.status is not ProcessingStatus.COMPLETED:
            continue
        text = " ".join(segment.text.split())
        if not text:
            continue
        rows.append({
            "embedding_index": len(texts),
            "segment_id": segment.segment_id,
            "video_id": segment.video_id,
            "segment_index": segment.segment_index,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "language": segment.language,
            "confidence": segment.confidence,
        })
        texts.append(text)
    if not texts:
        raise ValueError("Transcript artifact contains no usable completed segments")
    return texts, pd.DataFrame(rows)
```

- [ ] **Step 4: Encode with the same BGE-M3 adapter used for Context and build `SegmentDenseIndex`**

Use the existing normalized text encoding helper or move that pure helper into a small shared text module if importing it would create a cycle. Do not prepend timestamps/speaker/language to the embedded text in V1. If a remote adapter is injected, it must use remote source `"text"` so the hosted service selects the same BGE-compatible encoder as Context.

- [ ] **Step 5: Add service boundary and integrity lineage**

`RetrievalService.build_asr_segment_artifacts(...)` loads `TranscriptStore(transcripts_path)`, uses `models.resolved_evidence_embedding`, writes through a staging directory, validates `SegmentDenseIndex.load(staged)`, then atomically publishes.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_asr_segment_retriever.py tests/retrieval/test_segment_dense_index.py -q
pyright src/hcmai/retrieval/retriever/segment/artifacts.py src/hcmai/retrieval/retriever/pipeline.py

git add src/hcmai/retrieval/retriever/segment src/hcmai/retrieval/retriever/pipeline.py tests/retrieval/test_asr_segment_retriever.py
git commit -m "feat(retrieval): build asr segment index"
```

---

### Task 8: Project ASR Segment Hits to Canonical Frames Before RRF

**Files:**
- Modify: `src/hcmai/data/stores/frame.py`
- Create: `src/hcmai/retrieval/retriever/segment/projector.py`
- Create: `src/hcmai/retrieval/retriever/segment/retriever.py`
- Test: `tests/retrieval/test_segment_frame_projector.py`
- Test: `tests/retrieval/test_asr_segment_retriever.py`

**Interfaces:**
- Consumes: `FrameStore`, `SegmentDenseIndex`, BGE query batch.
- Produces: `SegmentFrameProjection`; `SegmentFrameProjector.project_row(row) -> SegmentFrameProjection | None`; `ASRSegmentRetriever.search_vectors(...) -> list[RetrievalResult]` where every emitted candidate has canonical `frame_id` and source `RetrievalSource.ASR`.

- [ ] **Step 1: Add deterministic per-video frame access to `FrameStore`**

Test then implement:

```python
def get_by_video(self, video_id: str) -> tuple[FrameRecord, ...]:
    return self._records_by_video.get(video_id, ())
```

Return the existing `(timestamp_ms, frame_idx, frame_id)` sorted tuple; do not copy/re-sort per query.

- [ ] **Step 2: Write projector tests for in-span, midpoint fallback, gap rejection, and deterministic ties**

```python
def test_projector_prefers_frame_inside_segment(frame_store):
    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=5000)
    projection = projector.project("v1", start_ms=1000, end_ms=2000)
    assert projection.frame_id == "f_inside"
    assert projection.kind == "inside_segment"


def test_projector_rejects_far_midpoint_fallback(frame_store):
    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=100)
    assert projector.project("v1", start_ms=9000, end_ms=10000) is None
```

For multiple frames inside a segment, select the one minimizing `abs(frame.timestamp_ms - midpoint_ms)`, then tie-break by `(timestamp_ms, frame_idx, frame_id)`.

- [ ] **Step 3: Implement `SegmentFrameProjector`**

```python
@dataclass(frozen=True, slots=True)
class SegmentFrameProjection:
    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    distance_ms: int
    kind: Literal["inside_segment", "nearest_midpoint"]
```

Algorithm:

```python
midpoint = (start_ms + end_ms) // 2
frames = self.frame_store.get_by_video(video_id)
inside = [f for f in frames if start_ms <= f.timestamp_ms < end_ms]
if inside:
    chosen = min(inside, key=lambda f: (abs(f.timestamp_ms-midpoint), f.timestamp_ms, f.frame_idx, f.frame_id))
    return projection(chosen, "inside_segment")
chosen = min(frames, key=lambda f: (abs(f.timestamp_ms-midpoint), f.timestamp_ms, f.frame_idx, f.frame_id))
distance = abs(chosen.timestamp_ms - midpoint)
return None if distance > self.max_projection_gap_ms else projection(chosen, "nearest_midpoint")
```

- [ ] **Step 4: Implement `ASRSegmentRetriever` with the same `source_family="text"` batch contract**

It must expose `encode()` via `encode_query_batch(..., source_family="text")`, call `SegmentDenseIndex.search_filtered()`, project each returned segment row, and materialize canonical frame candidates.

When several segments project to one frame, keep the strongest raw segment score. After deduplication, sort by `(-score, frame_id)` and assign new one-based ASR ranks.

Candidate metadata must retain segment provenance:

```python
metadata={
    "frame": {
        "frame_id": projection.frame_id,
        "video_id": projection.video_id,
        "frame_idx": projection.frame_idx,
        "timestamp_ms": projection.timestamp_ms,
    },
    "asr_segment": {
        "segment_id": row["segment_id"],
        "start_ms": int(row["start_ms"]),
        "end_ms": int(row["end_ms"]),
        "projection_kind": projection.kind,
        "projection_distance_ms": projection.distance_ms,
        "segment_score": score,
    },
}
```

- [ ] **Step 5: Verify query embedding reuse compatibility**

Build a Context `DenseRetriever` and `ASRSegmentRetriever` with the same fake BGE encoder, put them in `RRFFusionRetriever`, call `search_batch(["q1", "q2"])`, and assert the fake encoder saw one text-family batch for both retrievers rather than one encode call per source.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_segment_frame_projector.py tests/retrieval/test_asr_segment_retriever.py -q
pyright src/hcmai/data/stores/frame.py src/hcmai/retrieval/retriever/segment

git add src/hcmai/data/stores/frame.py src/hcmai/retrieval/retriever/segment tests/retrieval/test_segment_frame_projector.py tests/retrieval/test_asr_segment_retriever.py
git commit -m "feat(retrieval): project asr segment hits to canonical frames"
```

---

### Task 9: Compose Visual + Context + ASR Segment Retrieval Without Changing RRF

**Files:**
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Test: `tests/retrieval/test_fast_track_retrieval_composition.py`

**Interfaces:**
- Consumes: `DenseIndex` Visual, `DenseIndex` Context, `SegmentDenseIndex` ASR, `FrameStore`, Visual query encoder, shared BGE query encoder, existing `FusionConfig`.
- Produces: `RetrievalService.from_fast_track_indexes(...) -> RetrievalService`; existing `RRFFusionRetriever` receives only frame-native candidates.

- [ ] **Step 1: Write failing composition test**

```python
def test_fast_track_service_returns_frame_native_fused_candidates(
    visual_index,
    context_index,
    segment_index,
    frame_store,
    fake_visual_encoder,
    fake_bge,
    fusion_config,
):
    service = RetrievalService.from_fast_track_indexes(
        visual_index=visual_index,
        visual_encoder=fake_visual_encoder,
        context_index=context_index,
        asr_segment_index=segment_index,
        text_encoder=fake_bge,
        frame_store=frame_store,
        fusion=fusion_config,
        max_projection_gap_ms=5000,
    )
    result = service.search("red cable car", top_k=10)
    assert result.candidates
    assert all(candidate.frame_id for candidate in result.candidates)
    assert RetrievalSource.CONTEXT in service.active_sources
    assert RetrievalSource.ASR in service.active_sources
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_fast_track_retrieval_composition.py -q
```

- [ ] **Step 3: Add a specific fast-track factory; preserve `from_indexes()` for rollback**

```python
@classmethod
def from_fast_track_indexes(
    cls,
    *,
    visual_index: DenseIndex,
    visual_encoder: TextEmbeddingAdapter,
    context_index: DenseIndex | None,
    asr_segment_index: SegmentDenseIndex | None,
    text_encoder: TextEmbeddingAdapter | None,
    frame_store: FrameStore,
    fusion: FusionConfig,
    cache_config: RetrievalCacheConfig | None = None,
    max_projection_gap_ms: int = 5000,
) -> "RetrievalService":
    ...
```

Construct:

```text
DenseRetriever(source=VISUAL)
ContextRetriever(source=CONTEXT)          if context index exists
ASRSegmentRetriever(source=ASR)           if segment index exists
```

All retrievers share one cache object. Context/ASR use the same text encoder and `source_family="text"`; Visual remains `source_family="visual"`.

- [ ] **Step 4: Do not change `RRFFusionRetriever._fuse()`**

The acceptance test is that ASR projection is complete before `RRFFusionRetriever` sees candidates. If implementation work appears to require a generic segment identity inside RRF, stop and revisit Task 8 instead of widening scope.

- [ ] **Step 5: Add regression test that `score_visual_videos()` still finds the visual retriever by `source_family="visual"`**

This protects TRAKE's current candidate-local visual path without importing/modifying TRAKE code.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_fast_track_retrieval_composition.py -q
pyright src/hcmai/retrieval/retriever/pipeline.py

git add src/hcmai/retrieval/retriever/pipeline.py tests/retrieval/test_fast_track_retrieval_composition.py
git commit -m "feat(retrieval): compose fast track multimodal search"
```

---

### Task 10: Wire the New Stores/Indexes Into Online Startup With an Explicit Rollback Profile

**Files:**
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/common/config.py`
- Modify: `configs/baseline.yaml`
- Test: `tests/orchestration/test_fast_track_setup.py`

**Interfaces:**
- Consumes: `IndexConfig.profile`, new context/transcript paths, Visual/Context/ASR-segment index dirs, `models.resolved_evidence_embedding`.
- Produces: default `context_asr_segment` startup; `HCMAI_RETRIEVAL_PROFILE=legacy_specialists` emergency rollback; no online rebuilding.

- [ ] **Step 1: Write failing startup tests for modern and rollback profiles**

```python
def test_default_setup_loads_visual_context_and_segment_asr(fake_artifacts, monkeypatch):
    monkeypatch.setenv("HCMAI_RETRIEVAL_PROFILE", "context_asr_segment")
    service = load_search_service([])
    assert service.retrieval.active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    )


def test_legacy_profile_keeps_specialist_loader(fake_legacy_artifacts, monkeypatch):
    monkeypatch.setenv("HCMAI_RETRIEVAL_PROFILE", "legacy_specialists")
    service = load_search_service([])
    assert RetrievalSource.CAPTION in service.retrieval.active_sources
```

Mock model adapters; tests must not load GPU models.

- [ ] **Step 2: Update `_load_data()` to load typed new stores at startup**

Call `DataService.load(...)` with:

```python
context_path=settings.dataset.enrichment.context_path,
transcript_path=settings.dataset.enrichment.transcripts_path,
```

only when those paths exist. Keep legacy `load_evidence()` loop only for the legacy profile.

- [ ] **Step 3: Add modern index paths and environment overrides**

```python
context_path = Path(os.getenv("HCMAI_CONTEXT_INDEX_PATH", str(settings.index.context_path)))
asr_segment_path = Path(os.getenv("HCMAI_ASR_SEGMENT_INDEX_PATH", str(settings.index.asr_segment_path)))
profile = os.getenv("HCMAI_RETRIEVAL_PROFILE", settings.index.profile)
```

- [ ] **Step 4: Split `_load_retrieval()` by profile without duplicating Visual load**

Visual load/encoder remains first and required. For modern profile:

1. load Context `DenseIndex` if available;
2. load `SegmentDenseIndex` if available;
3. verify both use `models.resolved_evidence_embedding.model_name` and compatible dimensions;
4. construct one text encoder;
5. call `RetrievalService.from_fast_track_indexes(...)`, passing `settings.index.asr_projection_max_gap_ms`.

For `legacy_specialists`, call the existing `_load_text_indexes()` + `RetrievalService.from_indexes()` path.

- [ ] **Step 5: Update required-source degradation semantics**

If `VISUAL` is unavailable, startup retrieval fails as today. If `CONTEXT` or `ASR` is in `required_sources` and its new bundle is unavailable/incompatible, return `None` with an explicit readiness message. If optional, continue with remaining sources and a warning.

- [ ] **Step 6: Make `configs/baseline.yaml` default to the modern profile**

Target shape:

```yaml
index:
  profile: context_asr_segment
  path: artifacts/indexes/visual
  context_path: artifacts/indexes/context
  asr_segment_path: artifacts/indexes/asr_segments

  # rollback-only
  caption_path: artifacts/indexes/caption
  ocr_path: artifacts/indexes/ocr
  asr_path: artifacts/indexes/asr

search:
  fusion:
    required_sources: [visual]
    task_weights:
      kis:   {visual: 1.0, context: 1.0, asr: 1.0, caption: 1.0, ocr: 1.0}
      vkis:  {visual: 1.0, context: 1.0, asr: 1.0, caption: 1.0, ocr: 1.0}
      vqa:   {visual: 1.0, context: 1.0, asr: 1.0, caption: 1.0, ocr: 1.0}
      trake: {visual: 1.0, context: 1.0, asr: 1.0, caption: 1.0, ocr: 1.0}
```

Weights remain neutral until measured. Inactive legacy sources do not contribute because RRF normalizes active weights.

- [ ] **Step 7: Run startup tests and type checking**

```bash
PYTHONPATH=.:src python -m pytest tests/orchestration/test_fast_track_setup.py tests/retrieval/test_fast_track_retrieval_composition.py -q
pyright src/hcmai/orchestration/setup.py src/hcmai/common/config.py
```

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/orchestration/setup.py src/hcmai/common/config.py configs/baseline.yaml tests/orchestration/test_fast_track_setup.py
git commit -m "feat(orchestration): enable fast track retrieval profile"
```

---

### Task 11: Add One Offline Build Command and the Local↔ThunderCompute A6000 Runbook

**Files:**
- Create: `scripts/build_retrieval_indexes.py`
- Create: `scripts/sync_thundercompute_indexes.sh`
- Create: `docs/runbooks/thundercompute-index-build.md`
- Modify: `scripts/README.md`
- Test: `tests/retrieval/test_fast_track_retrieval_composition.py` (CLI/config smoke through monkeypatching service builders)

**Interfaces:**
- Consumes: `configs/indexing.yaml`, `configs/indexing.models.yaml`, local/staged paths.
- Produces: commands `preflight|visual|context|asr|all|validate`; validated bundles in `artifacts/indexes/{visual,context,asr_segments}`; transfer script driven entirely by environment variables.

- [ ] **Step 1: Write a failing CLI orchestration test**

Monkeypatch the three build functions and assert `--stage all` calls exactly:

```text
preflight -> visual -> context -> asr -> validate
```

in that order. Assert `--stage context` does not invoke Visual or ASR.

- [ ] **Step 2: Implement `build_retrieval_indexes.py` as a thin orchestrator**

Arguments:

```text
--config configs/indexing.yaml
--model-config configs/indexing.models.yaml
--stage {preflight,visual,context,asr,all,validate}
--frames <optional override>
--keyframes-root <optional override>
--map-keyframes-root <optional override>
--context <optional override>
--transcripts <optional override>
--output-root <optional override>
```

`preflight` must verify:

```text
mapping video count == 873
mapping row count == 177321
canonical frame count == 177321
unique frame_id count == 177321
mapping↔(video_id,keyframe_order) join is complete
a portable frame projection produced by `project_keyframe_paths(...)` resolves every staged keyframe under the supplied keyframes root
FrameContext frame_ids are a subset of canonical frame_ids and lineage matches
TranscriptStore loads with unique segment_id and positive durations
```

Do not hard-fail on the known 614 duplicate `(video_id, frame_idx)` rows; report them.

- [ ] **Step 3: Implement sequential GPU stage orchestration**

`all` performs:

```python
projected_frames = run_preflight(...)  # writes/returns a temporary indexing-only frames projection
build_visual(frames_path=projected_frames, ...)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
build_context(...)
build_asr(...)
run_validate(...)
```

Do not keep a SigLIP model reference alive while BGE runs. Context and ASR may reuse one injected BGE adapter within the same process; if the current service boundary creates two adapters, explicitly instantiate one `EmbeddingService.create_text_adapter(...)` and pass it to both builders.

- [ ] **Step 4: Implement final validation**

`validate` loads all three published indexes and checks:

```text
Visual vector_count == 177321
Visual frame_id set == canonical frame_id set
Context frame_ids subset canonical frame_ids
ASR mapping has no frame_id column
all three metadata model/revision/dim/normalization contracts match their configured encoder
Context and ASR text index dimensions are equal
all v2 checksums verify through loader round-trip
```

Write `artifacts/indexes/build_report.json` containing counts, paths, models/revisions, sizes, and validation status.

- [ ] **Step 5: Create the parameterized sync script**

`scripts/sync_thundercompute_indexes.sh` begins with:

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${HCMAI_THUNDER_HOST:?set HCMAI_THUNDER_HOST, e.g. ssh alias or user@host}"
: "${HCMAI_THUNDER_ROOT:?set HCMAI_THUNDER_ROOT to the repo/data root on ThunderCompute}"
: "${HCMAI_LOCAL_ROOT:?set HCMAI_LOCAL_ROOT to the local HCMAI working root}"
```

Support subcommands:

```text
push-inputs
pull-indexes
```

`push-inputs` syncs only:

```text
frames.parquet + manifest.json
keyframes/
map_keyframes/
frame_context_v1.parquet + context manifest
transcripts/
configs/indexing.yaml
configs/indexing.models.yaml
source code required to run the builders
```

It must not upload the 100+ GB raw video corpus. `pull-indexes` syncs only `visual/`, `context/`, `asr_segments/`, and `build_report.json`.

- [ ] **Step 6: Write the ThunderCompute runbook with exact environment-driven commands**

The runbook uses:

```bash
export HCMAI_LOCAL_ROOT=/absolute/path/to/local/hcmai
export HCMAI_THUNDER_HOST=<your-existing-ssh-alias-or-user-at-host>
export HCMAI_THUNDER_ROOT=/absolute/path/on/thundercompute/hcmai
```

Because the actual SSH hostname/path are user-owned infrastructure values not present in the repository, the script enforces them at runtime rather than embedding guessed values.

Then document:

```bash
bash scripts/sync_thundercompute_indexes.sh push-inputs
ssh "$HCMAI_THUNDER_HOST"
cd "$HCMAI_THUNDER_ROOT"
python scripts/build_retrieval_indexes.py --stage preflight --config configs/indexing.yaml --model-config configs/indexing.models.yaml
nvidia-smi
python scripts/build_retrieval_indexes.py --stage visual --config configs/indexing.yaml --model-config configs/indexing.models.yaml
python scripts/build_retrieval_indexes.py --stage context --config configs/indexing.yaml --model-config configs/indexing.models.yaml
python scripts/build_retrieval_indexes.py --stage asr --config configs/indexing.yaml --model-config configs/indexing.models.yaml
python scripts/build_retrieval_indexes.py --stage validate --config configs/indexing.yaml --model-config configs/indexing.models.yaml
exit
bash scripts/sync_thundercompute_indexes.sh pull-indexes
```

Also document the faster single-process option after the individual stages have passed once:

```bash
python scripts/build_retrieval_indexes.py --stage all --config configs/indexing.yaml --model-config configs/indexing.models.yaml
```

- [ ] **Step 7: Run CLI tests/shell syntax check and commit**

```bash
PYTHONPATH=.:src python -m pytest tests/retrieval/test_fast_track_retrieval_composition.py -q
bash -n scripts/sync_thundercompute_indexes.sh
python scripts/build_retrieval_indexes.py --help

git add scripts/build_retrieval_indexes.py scripts/sync_thundercompute_indexes.sh scripts/README.md docs/runbooks/thundercompute-index-build.md
git commit -m "feat(ops): add thundercompute multimodal index workflow"
```

---

### Task 12: Run the Focused Fast-Track Regression Gate and Freeze Rollback

**Files:**
- Modify: `scripts/validate_repository.sh` only if the real repository gate has a focused retrieval section suitable for adding these tests.
- Test: all focused files created above.
- Documentation: `docs/runbooks/thundercompute-index-build.md` final promotion/rollback section.

**Interfaces:**
- Consumes: all implemented artifacts and runtime composition.
- Produces: B0/B1/B2 gate evidence; explicit rollback command; no claim of large-benchmark validation.

- [ ] **Step 1: Run the complete focused unit/integration suite**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/data/test_btc_keyframe_map.py \
  tests/retrieval/test_index_artifact_integrity.py \
  tests/retrieval/test_visual_embedding_resume.py \
  tests/retrieval/test_context_index.py \
  tests/retrieval/test_segment_dense_index.py \
  tests/retrieval/test_segment_frame_projector.py \
  tests/retrieval/test_asr_segment_retriever.py \
  tests/retrieval/test_fast_track_retrieval_composition.py \
  tests/orchestration/test_fast_track_setup.py \
  -q
```

Expected: PASS without loading hosted inference or real GPU models.

- [ ] **Step 2: Run static checks on every touched package**

```bash
pyright \
  src/hcmai/common \
  src/hcmai/data/ingestion \
  src/hcmai/data/stores/frame.py \
  src/hcmai/data/pipeline.py \
  src/hcmai/retrieval \
  src/hcmai/orchestration/setup.py \
  scripts/build_retrieval_indexes.py

git diff --check
```

- [ ] **Step 3: Verify the real full-corpus index bundles after pulling them back locally**

```bash
python scripts/build_retrieval_indexes.py \
  --stage validate \
  --config configs/indexing.yaml \
  --model-config configs/indexing.models.yaml
```

Expected `build_report.json`:

```text
visual.status = complete
visual.vector_count = 177321
context.status = complete
asr_segments.status = complete
validation.status = pass
```

Context and ASR counts are data-dependent and must equal their own usable source rows, not 177,321.

- [ ] **Step 4: B0 — Visual regression smoke**

Use several known queries that previously behaved well with Visual + Caption. Start with modern profile but temporarily disable optional Context/ASR by pointing their paths to absent directories or configuring them optional, then confirm Visual results resolve to valid canonical frame IDs and videos. Do not claim metric equivalence from qualitative inspection; record this as B0 smoke only.

- [ ] **Step 5: B1 — Visual + Context smoke**

Enable Context, keep ASR optional/off, run the same known queries, and confirm:

```text
no unknown frame_id
no dataset/model mismatch
RRF completes
TemporalEvidenceCore completes
KIS/VQA localization returns canonical scenes/frames
```

- [ ] **Step 6: B2 — Visual + Context + projected ASR end-to-end smoke**

Enable all three. Add at least one speech-heavy query and verify returned ASR candidate metadata contains `segment_id/start_ms/end_ms/projection_kind` while the outer candidate identity is canonical `frame_id`.

- [ ] **Step 7: Verify rollback is one environment/config switch**

Modern:

```bash
export HCMAI_RETRIEVAL_PROFILE=context_asr_segment
```

Emergency rollback:

```bash
export HCMAI_RETRIEVAL_PROFILE=legacy_specialists
```

Restart the service and confirm `active_sources` changes accordingly; no artifact regeneration occurs.

- [ ] **Step 8: Run the repository release gate available in the real working tree**

```bash
scripts/validate_repository.sh
```

If unrelated pre-existing failures remain, record them separately. Do not mask a failure in any task touched by this migration.

- [ ] **Step 9: Commit the final gate/docs change**

```bash
git add scripts/validate_repository.sh docs/runbooks/thundercompute-index-build.md
git commit -m "test(retrieval): gate fast track multimodal migration"
```

---

## Implementation Order and Stop Conditions

Execute strictly in this order:

```text
1 BTC mapping correctness
      ↓
2 index integrity/publication
      ↓
3 Visual strict/resumable rebuild ----------------------┐
      ↓                                                 │
4 source/config contracts                               │
      ↓                                                 │
5 FrameContext index                                    │
      ↓                                                 │
6 SegmentDenseIndex                                     │
      ↓                                                 │
7 ASR segment artifact                                  │
      ↓                                                 │
8 ASR projection/retriever                              │
      ↓                                                 │
9 fast-track retrieval composition                      │
      ↓                                                 │
10 online startup + rollback                            │
      ↓                                                 │
11 ThunderCompute build/sync orchestration <------------┘
      ↓
12 B0/B1/B2 regression gate
```

Stop rather than widening scope if any of these occur:

- implementing ASR requires changing `RetrievalCandidate.frame_id`;
- RRF starts receiving `segment_id` as identity;
- a change to `src/hcmai/temporal/**` appears necessary;
- TRAKE scoring changes as a side effect;
- a partial Visual corpus is about to be published as the default index;
- online startup attempts to rebuild a missing bundle.

---

## Minimal Competition-Time Priority Cut

If implementation time becomes critical, preserve correctness in this order:

```text
P0  Task 1 + 2 + 3
    Rebuild a correct complete Visual index first.

P1  Task 4 + 5 + 9 + 10
    Ship Visual + FrameContext online.
    This already replaces the proven Visual + Caption pattern with the new fused context channel.

P2  Task 6 + 7 + 8
    Add segment-native ASR and projection.

P3  Task 11 + 12
    Operational polish, full sync automation, and expanded smoke evidence.
```

If P2 cannot be completed safely before the competition deadline, run `Visual + Context` rather than falling back to frame-aligned ASR or introducing a rushed generic temporal-identity rewrite.

---

## Self-Review Results

### Spec coverage

Every fast-track scope item maps to a task: BTC mapping (1), Visual rebuild (3), generic text configuration (4), Context index (5), SegmentDenseIndex (6), ASR corpus (7), segment projection (8), online composition (9–10), immutable integrity/publication (2), ThunderCompute workflow (11), and minimal rollout gate (12). No task changes temporal core, reranker, or TRAKE.

### Placeholder scan

The plan contains no unresolved implementation placeholders or unspecified behavior. Infrastructure-specific SSH host/root values are intentionally runtime environment variables with shell validation because those values are not present in the repository and must not be guessed.

### Type consistency

- Frame-native indexes use `DenseIndex` and canonical `frame_id`.
- Segment-native ASR uses `SegmentDenseIndex` and `segment_id` until `ASRSegmentRetriever` projects to frame candidates.
- Context and ASR use one `TextEmbeddingAdapter`/BGE-M3 query family.
- `RetrievalSource.CONTEXT` is additive; legacy sources remain valid for rollback.
- `RetrievalService.from_fast_track_indexes()` is the supported online factory; the former `from_indexes()` rollback factory was removed by the P2 cleanup.
- No new method relies on a segment identity inside `RetrievalCandidate` or RRF.
