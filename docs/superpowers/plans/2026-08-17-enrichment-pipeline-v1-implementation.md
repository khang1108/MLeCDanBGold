# HCMAI Enrichment Pipeline V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the preprocessing-dependent enrichment path with a BTC-keyframe-native, experiment-friendly enrichment pipeline that independently materializes Caption, OCR, Object, and ASR evidence and deterministically builds `frame_context-v1 = Caption + normalized OCR + normalized Object summary`.

**Architecture:** BTC keyframes and BTC object JSON become canonical inputs. Expensive enrichers materialize specialist artifacts independently; Object import is non-destructive; ASR remains segment/timeline evidence; and a cheap deterministic context builder joins only completed Caption/OCR/Object evidence by canonical `frame_id`. Existing `FrameEnrichment` is retained only as a compatibility projection during the transition, never as the source of truth.

**Tech Stack:** Python 3.12+, Pydantic v2, pandas/Parquet, PIL, Hugging Face Transformers/Florence-2, Qwen ASR pipeline, existing `hcmai.common.utils.io.atomic_write`, pytest, pyright.

## Global Constraints

- BTC keyframes replace custom preprocessing/keyframe extraction for this competition profile.
- BTC objects are imported, not redetected.
- Caption/OCR/Object remain independently materialized specialist evidence.
- ASR remains an independent timeline modality and is not serialized into `frame_context-v1`.
- `frame_context-v1 = Caption + normalized OCR + normalized Object summary`.
- Frame context serialization is deterministic; no LLM/VLM rewrite, translation, sampling, or semantic inference is allowed.
- Visual embeddings are not inputs to frame context and remain an indexing concern.
- Temporal-context embedding, scene segmentation, learned multimodal fusion, and retrieval/index fusion are out of scope.
- Missing modalities are omitted from context; never serialize `None`, `No OCR`, `No objects`, or equivalent placeholders.
- Raw evidence is immutable source evidence; normalization/truncation may affect derived views only.
- Context section order is exactly `[CAPTION]`, `[VISIBLE_TEXT]`, `[OBJECTS]`.
- V1 context defaults: Caption <= 80 whitespace tokens; OCR <= 80; Objects <= 40; the independent section budgets cap context at 200 whitespace tokens.
- Individual frame/segment failures must not fail the whole corpus; completed rows are resumable and failed/incomplete rows are retryable.
- Changing one specialist artifact invalidates only its dependent views; changing ASR never invalidates `frame_context-v1`.
- The active HCMAI 2026 path is BTC-native; legacy custom-video preprocessing
  is removed, while the low-level `video.py` timing utility remains available.

---

## File Structure to Lock In

### Create

- `src/hcmai/common/schemas/evidence.py` — specialist evidence and `FrameContext` contracts.
- `src/hcmai/data/ingestion/__init__.py` — BTC ingestion public exports.
- `src/hcmai/data/ingestion/btc.py` — reusable BTC frame-store import previously embedded in the root script.
- `src/hcmai/data/enrichment/objects/__init__.py` — object enrichment public exports.
- `src/hcmai/data/enrichment/objects/config.py` — object import/summary policy.
- `src/hcmai/data/enrichment/objects/importer.py` — BTC OpenImages/Faster-RCNN JSON normalization.
- `src/hcmai/data/enrichment/objects/artifacts.py` — atomic object artifact writer/validator.
- `src/hcmai/data/enrichment/context/__init__.py` — context enrichment public exports.
- `src/hcmai/data/enrichment/context/config.py` — deterministic serializer budgets/version.
- `src/hcmai/data/enrichment/context/serializer.py` — pure normalization/budget/serialization logic.
- `src/hcmai/data/enrichment/context/builder.py` — artifact join, lineage validation, resume/build.
- `scripts/generate_object_enrichment.py` — thin CLI through `EnrichmentService`.
- `scripts/build_frame_context.py` — thin CLI through `EnrichmentService`.
- `tests/data/test_evidence_contracts.py`
- `tests/data/test_btc_ingestion.py`
- `tests/data/enrichment/test_caption_evidence.py`
- `tests/data/enrichment/test_ocr_evidence.py`
- `tests/data/enrichment/test_object_evidence.py`
- `tests/data/enrichment/test_frame_context.py`
- `tests/data/enrichment/test_asr_segment_evidence.py`
- `tests/data/test_data_service_evidence.py`
- `tests/data/enrichment/test_enrichment_v1_integration.py`

### Modify

- `src/hcmai/common/schemas/frame.py` — keep `FrameRecord`; downgrade `FrameEnrichment` to compatibility view and stop deduplicating repeated object labels.
- `src/hcmai/common/schemas/transcript.py` — add optional ASR provenance/confidence/status fields while preserving existing segment timing contract.
- `src/hcmai/common/schemas/inference.py` — carry structured OCR regions across remote inference.
- `src/hcmai/common/schemas/__init__.py` — export new contracts.
- `src/hcmai/data/pipeline.py` — BTC-native `prepare`, specialist stores, context lookup.
- `src/hcmai/data/stores/evidence.py` — load typed specialist artifacts instead of forcing everything through `FrameEnrichment`.
- `src/hcmai/data/stores/__init__.py` — export typed stores.
- `src/hcmai/data/enrichment/pipeline.py` — public facade methods for object import and context build.
- `src/hcmai/data/enrichment/caption/artifacts.py` — write `captions.parquet` as source of truth plus temporary legacy projection.
- `src/hcmai/data/enrichment/caption/resume.py` — resume from `CaptionEvidence`.
- `src/hcmai/data/enrichment/caption/runner.py` — produce `CaptionEvidence` with machine-readable failure codes and lineage.
- `src/hcmai/data/enrichment/caption/generator.py` — point resume/write/validation at `captions.parquet`.
- `src/hcmai/data/enrichment/ocr/models/entities.py` — region-aware OCR result contract.
- `src/hcmai/data/enrichment/ocr/adapters/florence.py` — use region OCR task and normalize boxes.
- `src/hcmai/data/enrichment/ocr/adapters/remote.py` — preserve remote OCR regions.
- `src/hcmai/data/enrichment/ocr/artifacts.py` — frame + region tables, normalization, quality score, temporary legacy projection.
- `src/hcmai/data/enrichment/ocr/generator.py` — typed OCR frame/region artifacts and resume.
- `src/hcmai/data/enrichment/ocr/report.py` — report normalized/raw/region coverage.
- `src/hcmai/llm/adapters/local.py` — return structured OCR results instead of flattening to strings.
- `src/hcmai/llm/server/api.py` — serialize OCR regions in hosted responses.
- `src/hcmai/data/enrichment/transcripts/adapters/asr.py` — stamp segment lineage/status.
- `src/hcmai/data/enrichment/transcripts/store.py` — load enriched transcript schema.
- `src/hcmai/data/enrichment/transcripts/materialize.py` — keep frame-aligned ASR only as a derived compatibility artifact.
- `scripts/ingest_btc_keyframes.py` — become a thin wrapper around `hcmai.data.ingestion.btc`.
- `scripts/prepare_data.py` — call the BTC-native preparation path with the correct arguments.
- `configs/enrichment.yaml` — add BTC input paths, object policy, OCR normalization, context policy/output paths.
- `configs/preparation.s3.yaml` — remove preprocessing from the HCMAI 2026 competition stage graph; add object/context stages without adding index/fusion work.
- `src/hcmai/data/README.md`
- `src/hcmai/data/enrichment/README.md`
- `scripts/README.md`

### Do Not Modify in This Plan

- `src/hcmai/data/preprocessing/video.py` — retained low-level video utility;
  legacy preprocessing modules are removed.
- `src/hcmai/retrieval/**`
- `src/hcmai/temporal/**`
- `src/hcmai/pipelines/kis/**`
- `src/hcmai/pipelines/vqa/**`
- `src/hcmai/pipelines/trake/**`

Those remain for the separate Indexing/Retrieval design and implementation.

---

### Task 1: Introduce Specialist Evidence Contracts Without Breaking `FrameRecord`

**Files:**

- Create: `src/hcmai/common/schemas/evidence.py`
- Modify: `src/hcmai/common/schemas/frame.py:40-61`
- Modify: `src/hcmai/common/schemas/__init__.py:1-133`
- Test: `tests/data/test_evidence_contracts.py`

**Interfaces:**

- Consumes: existing `ContractModel`, `NonEmptyString`, `ProcessingStatus`.
- Produces: `CaptionEvidence`, `OCRRegion`, `OCREvidence`, `ObjectDetection`, `ObjectEvidence`, `FrameContext`, and `usable_completed_text(...)` for later tasks.

- [X] **Step 1: Write failing schema tests for status, lineage, object multiplicity, and deterministic context lineage**

```python
from hcmai.common.schemas import (
    CaptionEvidence,
    FrameContext,
    ObjectDetection,
    ObjectEvidence,
    OCREvidence,
    OCRRegion,
    ProcessingStatus,
)


def test_object_evidence_preserves_repeated_instances():
    detections = [
        ObjectDetection(label="person", confidence=0.9, x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.4),
        ObjectDetection(label="person", confidence=0.8, x_min=0.3, y_min=0.1, x_max=0.4, y_max=0.4),
    ]
    row = ObjectEvidence(
        frame_id="L01_V001:0000",
        video_id="L01_V001",
        detections=detections,
        counts={"person": 2},
        summary="person x2",
        detection_count=2,
        artifact_version="object-v1",
        status=ProcessingStatus.COMPLETED,
    )
    assert row.counts["person"] == 2
    assert len(row.detections) == 2


def test_frame_context_records_upstream_versions():
    row = FrameContext(
        frame_id="f1",
        video_id="v1",
        context_text="[CAPTION]\nA person runs.",
        caption_text="A person runs.",
        ocr_text=None,
        object_summary=None,
        context_version="frame-context-v1",
        caption_version="caption-v1",
        ocr_version="ocr-v1",
        object_version="object-v1",
        frame_store_id="btc-v1",
    )
    assert row.context_version == "frame-context-v1"
```

- [X] **Step 2: Run the schema test and verify it fails because the contracts do not exist**

Run:

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/test_evidence_contracts.py -q
```

Expected: collection/import failure for the new evidence classes.

- [X] **Step 3: Add the new evidence contracts with explicit non-destructive fields**

Implement `src/hcmai/common/schemas/evidence.py` with these exact public shapes:

```python
class CaptionEvidence(ContractModel):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    text: str | None = None
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None


class OCRRegion(ContractModel):
    frame_id: NonEmptyString
    region_id: NonEmptyString
    region_order: int = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)


class OCREvidence(ContractModel):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    raw_text: str | None = None
    normalized_text: str | None = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    region_count: int = Field(default=0, ge=0)
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None


class ObjectDetection(ContractModel):
    label: NonEmptyString
    confidence: float = Field(ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)


class ObjectEvidence(ContractModel):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    detections: list[ObjectDetection] = Field(default_factory=list)
    counts: dict[NonEmptyString, int] = Field(default_factory=dict)
    summary: str | None = None
    detection_count: int = Field(default=0, ge=0)
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None


class FrameContext(ContractModel):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    caption_text: str | None = None
    ocr_text: str | None = None
    object_summary: str | None = None
    context_text: str | None = None
    caption_available: bool = False
    ocr_quality: float = Field(default=0.0, ge=0, le=1)
    object_count: int = Field(default=0, ge=0)
    context_version: NonEmptyString
    caption_version: NonEmptyString
    ocr_version: NonEmptyString
    object_version: NonEmptyString
    frame_store_id: NonEmptyString | None = None
```

Add model validators so `x_max >= x_min`, `y_max >= y_min`, `detection_count == len(detections)`, object `counts` exactly match `detections`, and failed rows require `error_code`/`error_message` while completed rows may carry empty evidence.

- [X] **Step 4: Convert `FrameEnrichment` into an explicitly legacy projection**

In `src/hcmai/common/schemas/frame.py`, keep the class for compatibility but:

```python
class FrameEnrichment(ContractModel):
    """Deprecated frame-aligned compatibility projection.

    Specialist artifacts in ``common.schemas.evidence`` are the source of truth.
    """
    ...
```

Delete the `deduplicate_objects` validator entirely. Do not add structured object fields here; repeated legacy labels must remain repeated rather than silently losing counts.

- [X] **Step 5: Export all new types through `hcmai.common.schemas`**

Add explicit imports and names to `__all__` so downstream code never imports enrichment-internal modules for common contracts.

- [X] **Step 6: Run focused tests and type checking**

Run:

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/test_evidence_contracts.py -q
pyright src/hcmai/common/schemas
```

Expected: PASS.

- [X] **Step 7: Commit the contract boundary**

```bash
git add src/hcmai/common/schemas tests/data/test_evidence_contracts.py
git commit -m "refactor(data): add specialist enrichment contracts"
```

---

### Task 2: Make BTC Keyframe Import the Competition Preparation Path

**Files:**

- Create: `src/hcmai/data/ingestion/__init__.py`
- Create: `src/hcmai/data/ingestion/btc.py`
- Modify: `src/hcmai/data/pipeline.py:74-94`
- Modify: `scripts/ingest_btc_keyframes.py:1-191`
- Modify: `scripts/prepare_data.py:13-35`
- Modify: `configs/enrichment.yaml:1-5`
- Test: `tests/data/test_btc_ingestion.py`

**Interfaces:**

- Consumes: BTC `metadata/frames.parquet`, keyframe paths, existing `FrameRecord`.
- Produces: `BTCIngestionConfig`, `import_btc_frame_store(config) -> Path`, canonical `frames.parquet`, `manifest.json`, and `DataService.prepare(config_path) -> Path` using BTC import for this profile.

- [X] **Step 1: Write a failing test that imports BTC metadata without touching preprocessing**

```python
def test_btc_import_does_not_require_preprocessing_fields(tmp_path):
    source = tmp_path / "btc"
    (source / "metadata").mkdir(parents=True)
    keyframe = source / "keyframes" / "L01_V001" / "0000.jpg"
    keyframe.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(keyframe)
    pd.DataFrame([{
        "frame_id": "L01_V001:0000",
        "video_id": "L01_V001",
        "frame_idx": 120,
        "keyframe_order": 1,
        "timestamp_ms": 4000,
        "image_path": "keyframes/L01_V001/0000.jpg",
        "width": 32,
        "height": 24,
    }]).to_parquet(source / "metadata" / "frames.parquet", index=False)

    output = import_btc_frame_store(BTCIngestionConfig(
        btc_root=source,
        data_root=source,
        output_root=tmp_path / "frame_store",
        frame_store_id="btc-test-v1",
    ))
    row = pd.read_parquet(output).iloc[0]
    assert row["shot_id"] is None
    assert row["event_id"] is None
    assert row["frame_id"] == "L01_V001:0000"
```

Monkeypatch `hcmai.data.preprocessing` imports to raise if touched, and assert the import still succeeds.

- [X] **Step 2: Run the test and verify it fails because BTC ingestion is still script-local**

Run:

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/test_btc_ingestion.py -q
```

Expected: import failure for `hcmai.data.ingestion.btc`.

- [X] **Step 3: Move script logic into a reusable ingestion module**

Create:

```python
@dataclass(frozen=True)
class BTCIngestionConfig:
    btc_root: Path
    data_root: Path
    output_root: Path
    frame_store_id: str


def import_btc_frame_store(config: BTCIngestionConfig) -> Path:
    ...
    return config.output_root / "frames.parquet"
```

Keep the existing median FPS estimation and atomic Parquet write. Preserve `shot_id=None`, `event_id=None`, preprocessing scores as neutral defaults only for schema compatibility, and `selection_reasons=("btc_keyframe",)`.

Write `manifest.json` with at least:

```json
{
  "pipeline_version": "btc-keyframe-ingestion-v1",
  "source": "btc_provided_keyframes",
  "frame_store_id": "...",
  "video_count": 0,
  "frame_count": 0
}
```

- [X] **Step 4: Make `scripts/ingest_btc_keyframes.py` a thin CLI**

The root script should only parse args, construct `BTCIngestionConfig`, call `import_btc_frame_store`, print the returned path, and exit. Remove duplicate ingestion/business logic from the script.

- [X] **Step 5: Redirect `DataService.prepare()` away from video preprocessing for this competition profile**

Replace the preprocessing import block with configuration loading from the `dataset` mapping in `configs/enrichment.yaml`. Required keys:

```yaml
dataset:
  version: "hcmai2026_v1"
  source: "btc_keyframes"
  btc_root: "data"
  data_root: "data"
  frame_store_id: "btc-keyframes-v1"
  frames_path: "artifacts/frame_store/frames.parquet"
  frame_store_output: "artifacts/frame_store"
```

`DataService.prepare(config_path, ...)` must reject any `dataset.source` other than `btc_keyframes` in this competition configuration and call `import_btc_frame_store`. The legacy custom-video entry point is removed.

- [X] **Step 6: Fix `scripts/prepare_data.py` to call the actual `DataService.prepare(config_path)` signature**

Use:

```python
output = DataService.prepare(args.config)
data = DataService.load(output, dataset_root=args.dataset_root)
```

Change CLI args to `--config` and optional `--dataset-root`; remove the currently invalid positional use `DataService.prepare(args.dataset_root, args.output)`.

- [X] **Step 7: Run ingestion tests**

Run:

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/test_btc_ingestion.py -q
```

Expected: PASS and no preprocessing import.

- [X] **Step 8: Commit BTC-native preparation**

```bash
git add src/hcmai/data/ingestion src/hcmai/data/pipeline.py scripts/ingest_btc_keyframes.py scripts/prepare_data.py configs/enrichment.yaml tests/data/test_btc_ingestion.py
git commit -m "feat(data): make BTC keyframes canonical input"
```

---

### Task 3: Migrate Caption Generation to `CaptionEvidence`

**Files:**

- Modify: `src/hcmai/data/enrichment/caption/artifacts.py:23-73`
- Modify: `src/hcmai/data/enrichment/caption/resume.py:23-96`
- Modify: `src/hcmai/data/enrichment/caption/runner.py:25-114`
- Modify: `src/hcmai/data/enrichment/caption/generator.py`
- Modify: `src/hcmai/data/enrichment/caption/report.py`
- Test: `tests/data/enrichment/test_caption_evidence.py`

**Interfaces:**

- Consumes: canonical frame rows and `CaptionAdapter.caption_batch(images)`.
- Produces: `artifacts/enrichment/captions/captions.parquet` containing `CaptionEvidence`; temporary compatibility `frame_enrichment.parquet` may be derived from it but must never drive resume/source-of-truth behavior.

- [X] **Step 1: Write failing tests for completed, failed, empty, and resumed caption rows**

```python
def test_caption_artifact_keeps_video_id_and_failure_code(tmp_path, fake_frames):
    adapter = FakeCaptionAdapter(["A person runs.", RuntimeError("oom")])
    report = generate_captions(..., adapter=adapter)
    table = pd.read_parquet(tmp_path / "captions.parquet")
    assert table.loc[0, "video_id"] == "v1"
    assert table.loc[0, "text"] == "A person runs."
    assert table.loc[1, "status"] == "failed"
    assert table.loc[1, "error_code"] == "RuntimeError"
```

Add a rerun assertion that the completed frame is skipped while the failed frame is retried.

- [X] **Step 2: Run the test and verify current code writes `FrameEnrichment` instead**

Run:

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_caption_evidence.py -q
```

Expected: FAIL because `captions.parquet` and typed fields do not exist.

- [X] **Step 3: Change runner rows from `FrameEnrichment` to `CaptionEvidence`**

The successful row must include the frame's `video_id`, not just `frame_id`:

```python
CaptionEvidence(
    frame_id=frame_id,
    video_id=video_id,
    text=str(result).strip(),
    frame_store_id=frame_store_id,
    artifact_version=config.enrichment_version,
    model_name=config.model_checkpoint,
    model_revision=resolved_revision,
    status=ProcessingStatus.COMPLETED,
)
```

Failures use:

```python
error_code=type(error).__name__[:100]
error_message=message
status=ProcessingStatus.FAILED
```

An empty model result is a failed row with `error_code="EmptyCaption"`; do not drop the frame identity.

- [X] **Step 4: Resume from `captions.parquet`, not the legacy projection**

`resume_rows(...)` should reuse only rows where:

```python
row.status == ProcessingStatus.COMPLETED
and row.artifact_version == config.enrichment_version
and row.model_name == config.model_checkpoint
and (frame_store_id is None or row.frame_store_id == frame_store_id)
and bool((row.text or "").strip())
```

Failed/malformed rows go back to `todo`.

- [X] **Step 5: Write source-of-truth plus compatibility projection atomically**

`write_caption_artifacts` must atomically write:

```text
captions.parquet               # source of truth
failures.json
frame_enrichment.parquet       # temporary derived compatibility projection
```

Build the legacy projection only from `CaptionEvidence`; never read it back for resume. Preserve repeated object semantics by leaving `objects=[]` in the caption-only compatibility table.

- [X] **Step 6: Update report names/metadata**

Set:

```json
{
  "artifact_version": "caption-evidence.v1",
  "source_artifact": "captions.parquet"
}
```

Keep prior latency, model revision, skipped/retried counts.

- [X] **Step 7: Run focused tests**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_caption_evidence.py -q
pyright src/hcmai/data/enrichment/caption
```

Expected: PASS.

- [X] **Step 8: Commit caption migration**

```bash
git add src/hcmai/data/enrichment/caption tests/data/enrichment/test_caption_evidence.py
git commit -m "refactor(enrichment): materialize caption evidence independently"
```

---

### Task 4: Upgrade OCR to Preserve Raw Regions and Derived Normalized Text

**Files:**

- Modify: `src/hcmai/data/enrichment/ocr/models/entities.py:15-21`
- Modify: `src/hcmai/data/enrichment/ocr/models/contracts.py:14-21`
- Modify: `src/hcmai/data/enrichment/ocr/adapters/florence.py:62-96`
- Modify: `src/hcmai/data/enrichment/ocr/adapters/remote.py:41-54`
- Modify: `src/hcmai/common/schemas/inference.py:140-154`
- Modify: `src/hcmai/llm/adapters/local.py:237-241`
- Modify: `src/hcmai/llm/server/api.py:220-247`
- Modify: `src/hcmai/data/enrichment/ocr/config.py:10-31`
- Modify: `src/hcmai/data/enrichment/ocr/artifacts.py:29-134`
- Modify: `src/hcmai/data/enrichment/ocr/generator.py:39-185`
- Modify: `src/hcmai/data/enrichment/ocr/report.py:24-116`
- Test: `tests/data/enrichment/test_ocr_evidence.py`

**Interfaces:**

- Consumes: images and `OCRAdapter.recognize_batch`.
- Produces: `ocr/frames.parquet` (`OCREvidence`), `ocr/regions.parquet` (`OCRRegion`), failures, report, and temporary frame-enrichment compatibility projection.

- [X] **Step 1: Write failing normalization/region tests**

```python
def test_ocr_keeps_raw_regions_but_filters_context_text():
    regions = [
        OCRRegion(frame_id="f1", region_id="f1:0", region_order=0,
                  text="AIC 2026", confidence=None,
                  x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.2),
        OCRRegion(frame_id="f1", region_id="f1:1", region_order=1,
                  text="||||", confidence=None,
                  x_min=0.1, y_min=0.3, x_max=0.2, y_max=0.4),
        OCRRegion(frame_id="f1", region_id="f1:2", region_order=2,
                  text="AIC 2026", confidence=None,
                  x_min=0.1, y_min=0.5, x_max=0.4, y_max=0.6),
    ]
    normalized = normalize_regions(regions, min_confidence=0.0)
    assert normalized.text == "AIC 2026"
    assert len(regions) == 3
    assert 0.0 <= normalized.quality_score <= 1.0
```

Also test exact row/region ordering after Parquet round-trip and retry of failed rows.

- [X] **Step 2: Run the OCR test and verify it fails because the adapter only returns one string**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_ocr_evidence.py -q
```

Expected: FAIL.

- [X] **Step 3: Extend `OCRResult` to carry structured regions**

Use a feature-local immutable entity:

```python
@dataclass(frozen=True)
class OCRRegionResult:
    text: str
    confidence: float | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class OCRResult:
    text: str
    regions: tuple[OCRRegionResult, ...] = ()
    raw_output: object | None = None
```

- [X] **Step 4: Switch Florence to region OCR and normalize quadrilaterals to axis-aligned [0,1] boxes**

Use Florence task `<OCR_WITH_REGION>` in both processor input and `post_process_generation`. Parse the returned `labels` and `quad_boxes`; for each quadrilateral compute:

```python
x_min = min(xs) / image.width
x_max = max(xs) / image.width
y_min = min(ys) / image.height
y_max = max(ys) / image.height
```

Clamp each value to `[0.0, 1.0]`. Florence has no calibrated confidence in the current adapter, so store `confidence=None` rather than inventing scores. Build `OCRResult.text` by joining region labels in returned order.

- [X] **Step 5: Carry OCR regions through hosted inference**

Add `OCRRegionItem` to `common.schemas.inference` and `regions: list[OCRRegionItem]` to `OCRItem`. Change `LocalInferenceRuntime.ocr()` to return the structured adapter results; change the API endpoint to serialize `text`, `raw_output`, and regions rather than `str(value)`. Change `RemoteOCRAdapter` to reconstruct `OCRRegionResult` exactly.

- [X] **Step 6: Implement deterministic OCR normalization**

Add config fields:

```python
min_region_confidence: float = 0.0
min_context_quality: float = 0.5
artifact_version: str = "ocr-v1"
```

Normalization rules, in this exact order:

1. NFC normalize each region string.
2. Collapse internal whitespace and trim.
3. If confidence exists and is below `min_region_confidence`, exclude that region from `normalized_text` only; retain it in `regions.parquet`.
4. Exclude a line from `normalized_text` if it contains no Unicode alphanumeric character (`any(ch.isalnum() for ch in text)` is false).
5. Deduplicate normalized lines case-insensitively while preserving first occurrence/order.
6. Join retained lines with `\n`.

Define quality deterministically:

```python
usable_ratio = usable_region_count / max(1, raw_region_count)
mean_confidence = mean(confidences) if confidences else 1.0
quality_score = min(1.0, usable_ratio * mean_confidence)
```

No text yields `quality_score=0.0` and a completed empty-evidence row.

- [X] **Step 7: Write separate frame and region artifacts**

Atomically write:

```text
ocr/frames.parquet
ocr/regions.parquet
ocr/failures.json
ocr/manifest.json
ocr/frame_enrichment.parquet   # temporary projection only
```

`raw_text` is the ordered join of all non-empty raw region texts before filtering; `normalized_text` is the derived view. `region_count` equals all preserved regions, not only context-usable regions.

- [X] **Step 8: Resume only from `ocr/frames.parquet` and verify region consistency**

A reusable frame row requires matching artifact/model/frame-store lineage and completed status. If a reusable frame reports `region_count > 0` but its region rows are missing or duplicated, mark it non-reusable and retry the frame.

- [X] **Step 9: Update OCR report**

Include:

```text
frames_with_raw_text
frames_with_normalized_text
frames_with_regions
raw_region_count
usable_region_count
mean_quality_score
```

Do not call text coverage “accuracy”.

- [X] **Step 10: Run tests and type checking**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_ocr_evidence.py -q
pyright src/hcmai/data/enrichment/ocr src/hcmai/common/schemas/inference.py src/hcmai/llm
```

Expected: PASS.

- [X] **Step 11: Commit OCR structure preservation**

```bash
git add src/hcmai/data/enrichment/ocr src/hcmai/common/schemas/inference.py src/hcmai/llm tests/data/enrichment/test_ocr_evidence.py
git commit -m "feat(enrichment): preserve structured OCR evidence"
```

---

### Task 5: Import BTC Objects as Structured Evidence

**Files:**

- Create: `src/hcmai/data/enrichment/objects/__init__.py`
- Create: `src/hcmai/data/enrichment/objects/config.py`
- Create: `src/hcmai/data/enrichment/objects/importer.py`
- Create: `src/hcmai/data/enrichment/objects/artifacts.py`
- Modify: `src/hcmai/data/enrichment/pipeline.py:24-63`
- Create: `scripts/generate_object_enrichment.py`
- Test: `tests/data/enrichment/test_object_evidence.py`

**Interfaces:**

- Consumes: canonical `frames.parquet` and one BTC object JSON per keyframe.
- Produces: `objects/frames.parquet`, `objects/detections.parquet`, `objects/manifest.json`, `ObjectEvidence` rows keyed 1:1 with canonical frames.

- [X] **Step 1: Write a failing object import test using the TensorFlow/OpenImages detector output shape**

```python
def test_btc_object_import_preserves_counts_and_boxes(tmp_path, frame_store):
    payload = {
        "detection_class_entities": ["Person", "Person", "Car"],
        "detection_scores": [0.91, 0.83, 0.72],
        "detection_boxes": [
            [0.10, 0.10, 0.50, 0.30],  # ymin, xmin, ymax, xmax
            [0.15, 0.40, 0.55, 0.60],
            [0.20, 0.65, 0.70, 0.95],
        ],
    }
    ...
    row = pd.read_parquet(output / "frames.parquet").iloc[0]
    assert json.loads(row["counts_json"]) == {"car": 1, "person": 2}
    assert row["summary"] == "person x2; car x1"
    assert len(pd.read_parquet(output / "detections.parquet")) == 3
```

The adapter is intentionally strict about the expected BTC/TensorFlow-array shape; malformed JSON must fail that frame with a structured error rather than silently guessing another schema.

- [X] **Step 2: Run the test and verify the object module does not exist**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_object_evidence.py -q
```

Expected: import failure.

- [X] **Step 3: Add object policy configuration**

```python
@dataclass(frozen=True)
class ObjectConfig:
    objects_root: Path
    output_dir: Path
    artifact_version: str = "object-v1"
    summary_min_confidence: float = 0.25
    max_summary_labels: int = 20
```

`summary_min_confidence` affects only derived counts/summary. Every valid raw detection is retained in `detections.parquet`.

- [X] **Step 4: Implement strict BTC JSON normalization**

For each canonical frame, derive the object file path from `video_id` plus keyframe stem/order using the same naming convention as BTC keyframes. Parse the three parallel arrays:

```text
detection_class_entities
detection_scores
detection_boxes  # [ymin, xmin, ymax, xmax], normalized
```

Requirements:

- arrays must have identical length;
- labels are NFC normalized, whitespace-collapsed, case-folded to canonical lowercase for counts;
- scores must be finite in `[0,1]`;
- boxes must contain four finite normalized values and satisfy min <= max;
- repeated labels create repeated `ObjectDetection` rows.

- [X] **Step 5: Build deterministic frame summaries without spatial reasoning**

For detections with `confidence >= summary_min_confidence`, count by normalized label. Sort summary entries by:

```python
(-count, -max_confidence_for_label, label)
```

Serialize at most `max_summary_labels` as:

```text
person x3; car x1; microphone x1
```

Do not generate sentences such as “three people stand left of a car”.

- [X] **Step 6: Write frame/detection artifacts and preserve partial failures**

Every canonical frame gets exactly one `ObjectEvidence` row. Missing/malformed JSON becomes `status=FAILED`, `error_code` and bounded `error_message`; it does not abort the corpus. Valid files with zero detections are `COMPLETED` with empty counts/summary.

Write detections as flat rows including `frame_id`/`video_id`; reject duplicate detection identity generated from the same `frame_id + detection_index`.

- [X] **Step 7: Expose object import through `EnrichmentService` and a thin CLI**

Add:

```python
@staticmethod
def import_objects(
    frames_path: str | Path,
    objects_root: str | Path,
    output_dir: str | Path,
    config: ObjectConfig,
    *,
    frame_store_id: str | None = None,
) -> dict[str, Any]: ...
```

The root script only parses arguments/config and calls this method.

- [X] **Step 8: Run tests**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_object_evidence.py -q
pyright src/hcmai/data/enrichment/objects
```

Expected: PASS.

- [X] **Step 9: Commit object evidence support**

```bash
git add src/hcmai/data/enrichment/objects src/hcmai/data/enrichment/pipeline.py scripts/generate_object_enrichment.py tests/data/enrichment/test_object_evidence.py
git commit -m "feat(enrichment): import structured BTC object evidence"
```

---

### Task 6: Keep ASR Segment-Native and Add Explicit Provenance

**Files:**

- Modify: `src/hcmai/common/schemas/transcript.py:8-26`
- Modify: `src/hcmai/data/enrichment/transcripts/adapters/asr.py:223-260`
- Modify: `src/hcmai/data/enrichment/transcripts/store.py:22-113`
- Modify: `src/hcmai/data/enrichment/transcripts/materialize.py:49-212`
- Modify: `src/hcmai/data/enrichment/transcripts/manifest.py`
- Test: `tests/data/enrichment/test_asr_segment_evidence.py`

**Interfaces:**

- Consumes: existing video/audio ASR flow.
- Produces: enriched `TranscriptSegment` rows that remain timeline-native; optional frame-aligned ASR remains a derived compatibility artifact only.

- [X] **Step 1: Write failing tests for segment provenance and no dependency from FrameContext**

```python
def test_transcript_segment_carries_asr_lineage():
    row = TranscriptSegment(
        segment_id="v1:0",
        video_id="v1",
        segment_index=0,
        start_ms=1000,
        end_ms=2000,
        text="hello",
        language="en",
        status=ProcessingStatus.COMPLETED,
        model_name="Qwen/Qwen3-ASR-1.7B-hf",
        model_revision="abc",
        artifact_version="asr-segment-v1",
    )
    assert row.model_name.startswith("Qwen/")
```

- [X] **Step 2: Run the test and verify the current schema lacks those fields**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_asr_segment_evidence.py -q
```

Expected: FAIL.

- [X] **Step 3: Extend `TranscriptSegment` backwards-compatibly**

Add optional/defaulted fields:

```python
confidence: float | None = Field(default=None, ge=0, le=1)
status: ProcessingStatus = ProcessingStatus.COMPLETED
model_name: NonEmptyString | None = None
model_revision: NonEmptyString | None = None
artifact_version: NonEmptyString = "asr-segment-v1"
error_code: NonEmptyString | None = None
error_message: NonEmptyString | None = None
```

Do not weaken the current positive-duration validator.

- [X] **Step 4: Stamp ASR model identity at segment creation**

When `ASRAdapter.transcribe_audio()` creates segments, set `model_name=self.config.model_name`, `model_revision=self.resolved_revision`, and `artifact_version="asr-segment-v1"`. If the backend provides no calibrated confidence, leave it `None`.

- [X] **Step 5: Keep frame-aligned ASR as a compatibility/materialized view only**

Do not remove `materialize_asr_enrichment` yet because current online retrieval expects it. Add module/class docstrings and manifest metadata marking it as:

```text
source_of_truth = transcript segments
frame_alignment = derived compatibility view
context_dependency = none
```

No FrameContext code may import `materialize_asr_enrichment` or `ASRStore`.

- [X] **Step 6: Run transcript tests**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_asr_segment_evidence.py -q
pyright src/hcmai/common/schemas/transcript.py src/hcmai/data/enrichment/transcripts
```

Expected: PASS.

- [X] **Step 7: Commit segment-native ASR lineage**

```bash
git add src/hcmai/common/schemas/transcript.py src/hcmai/data/enrichment/transcripts tests/data/enrichment/test_asr_segment_evidence.py
git commit -m "refactor(enrichment): keep ASR as versioned timeline evidence"
```

---

### Task 7: Implement Deterministic `frame_context-v1`

**Files:**

- Create: `src/hcmai/data/enrichment/context/__init__.py`
- Create: `src/hcmai/data/enrichment/context/config.py`
- Create: `src/hcmai/data/enrichment/context/serializer.py`
- Create: `src/hcmai/data/enrichment/context/builder.py`
- Modify: `src/hcmai/data/enrichment/pipeline.py`
- Create: `scripts/build_frame_context.py`
- Test: `tests/data/enrichment/test_frame_context.py`

**Interfaces:**

- Consumes: canonical frames, `captions.parquet`, `ocr/frames.parquet`, `objects/frames.parquet`.
- Produces: `context/frame_context_v1.parquet` and manifest. No model inference.

- [X] **Step 1: Write failing pure serializer tests first**

```python
def test_context_section_order_and_missing_omission():
    text = serialize_frame_context(
        caption="A man speaks.",
        ocr="AIC 2026",
        objects="person x2; microphone x1",
        config=FrameContextConfig(),
    )
    assert text == (
        "[CAPTION]\nA man speaks.\n\n"
        "[VISIBLE_TEXT]\nAIC 2026\n\n"
        "[OBJECTS]\nperson x2; microphone x1"
    )


def test_context_is_byte_deterministic():
    a = serialize_frame_context(...)
    b = serialize_frame_context(...)
    assert a.encode("utf-8") == b.encode("utf-8")


def test_context_does_not_include_asr():
    assert "SPEECH" not in serialize_frame_context(...)
```

Also test caption-only, OCR-only, object-only, all-missing, and exact 80/80/40 whitespace-token truncation.

- [X] **Step 2: Run the test and verify context module is missing**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_frame_context.py -q
```

Expected: import failure.

- [X] **Step 3: Add frozen V1 context configuration**

```python
@dataclass(frozen=True)
class FrameContextConfig:
    context_version: str = "frame-context-v1"
    caption_token_budget: int = 80
    ocr_token_budget: int = 80
    object_token_budget: int = 40
    min_ocr_quality: float = 0.5
```

Validate all budgets are non-negative and `min_ocr_quality` is in `[0,1]`.

- [X] **Step 4: Implement the pure serializer**

Use only deterministic string transforms:

```python
def _truncate_whitespace_tokens(text: str, limit: int) -> str:
    return " ".join(text.split()[:limit])


def serialize_frame_context(*, caption, ocr, objects, config) -> str | None:
    parts: list[str] = []
    if caption:
        parts.append("[CAPTION]\n" + _truncate_whitespace_tokens(caption, config.caption_token_budget))
    if ocr:
        parts.append("[VISIBLE_TEXT]\n" + _truncate_whitespace_tokens(ocr, config.ocr_token_budget))
    if objects:
        parts.append("[OBJECTS]\n" + _truncate_whitespace_tokens(objects, config.object_token_budget))
    return "\n\n".join(parts) or None
```

Do not import transformers, LLM services, visual embeddings, transcript stores, or ASR modules.

- [X] **Step 5: Implement artifact join with strict lineage checks**

For each canonical frame:

- caption is usable only when `CaptionEvidence.status == COMPLETED`, no error, and text non-empty;
- OCR is usable only when completed, normalized text non-empty, and `quality_score >= min_ocr_quality`;
- object summary is usable only when completed and non-empty;
- missing/failed sources are omitted, not fatal;
- `FrameContext.frame_store_id` must match all present upstream rows;
- `caption_version`, `ocr_version`, and `object_version` are copied from the upstream artifact manifests/rows.

Reject duplicate frame rows in any specialist input before joining.

- [X] **Step 6: Implement dependency-aware resume/invalidation**

Write `context/manifest.json` containing:

```json
{
  "context_version": "frame-context-v1",
  "caption_version": "...",
  "ocr_version": "...",
  "object_version": "...",
  "frame_store_id": "...",
  "serializer_config": {
    "caption_token_budget": 80,
    "ocr_token_budget": 80,
    "object_token_budget": 40,
    "min_ocr_quality": 0.5
  }
}
```

If this identity matches and `frame_context_v1.parquet` validates against canonical frame IDs, return it without rebuilding. If serializer config/version or any Caption/OCR/Object version changes, rebuild only the context artifact. ASR version is deliberately absent from this identity.

- [X] **Step 7: Expose builder through `EnrichmentService` and thin CLI**

Add:

```python
@staticmethod
def build_frame_context(
    frames_path: str | Path,
    caption_path: str | Path,
    ocr_frames_path: str | Path,
    object_frames_path: str | Path,
    output_dir: str | Path,
    config: FrameContextConfig,
    *,
    frame_store_id: str | None = None,
) -> Path: ...
```

- [X] **Step 8: Run serializer/builder tests and type checking**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_frame_context.py -q
pyright src/hcmai/data/enrichment/context
```

Expected: PASS.

- [X] **Step 9: Commit deterministic context build**

```bash
git add src/hcmai/data/enrichment/context src/hcmai/data/enrichment/pipeline.py scripts/build_frame_context.py tests/data/enrichment/test_frame_context.py
git commit -m "feat(enrichment): build deterministic frame context"
```

---

### Task 8: Make `DataService` Load Typed Specialist Evidence

**Files:**

- Modify: `src/hcmai/data/stores/evidence.py:20-139`
- Modify: `src/hcmai/data/stores/__init__.py:10-13`
- Modify: `src/hcmai/data/pipeline.py:16-171`
- Test: `tests/data/test_data_service_evidence.py`

**Interfaces:**

- Consumes: specialist Parquet artifacts and canonical frames.
- Produces: `CaptionStore`, `OCRStore`, `ObjectStore`, `FrameContextStore`, existing `ASRStore` compatibility; `DataService.get_evidence(...)` remains text-oriented for current downstream consumers while typed getters expose structured evidence.

- [X] **Step 1: Write failing store tests**

```python
def test_data_service_exposes_context_and_structured_objects(tmp_path, artifacts):
    data = DataService.load(
        artifacts.frames,
        evidence_paths={
            RetrievalSource.CAPTION: artifacts.caption,
            RetrievalSource.OCR: artifacts.ocr_frames,
        },
        object_path=artifacts.object_frames,
        context_path=artifacts.context,
    )
    assert data.get_evidence("f1", RetrievalSource.CAPTION) == "A person runs."
    assert data.get_object_evidence("f1").counts["person"] == 2
    assert data.get_frame_context("f1").context_version == "frame-context-v1"
```

Do not add `OBJECT`/`CONTEXT` to `RetrievalSource` in this plan; that belongs to Indexing/Retrieval design. Typed data access is enough here.

- [X] **Step 2: Run the test and verify object/context stores are absent**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/test_data_service_evidence.py -q
```

Expected: FAIL.

- [X] **Step 3: Replace `_TextEvidenceStore`'s universal `FrameEnrichment` materialization with typed store classes**

Use a small generic validated-Parquet loader internally, but public stores return their exact contracts:

```python
class CaptionStore: get(frame_id) -> CaptionEvidence
class OCRStore: get(frame_id) -> OCREvidence
class ObjectStore: get(frame_id) -> ObjectEvidence
class FrameContextStore: get(frame_id) -> FrameContext
```

Keep `get_text(frame_id)` on Caption/OCR/FrameContext stores for compatibility:

```text
Caption -> text
OCR -> normalized_text
FrameContext -> context_text
```

`ASRStore` may continue loading the derived frame-aligned compatibility artifact until the retrieval migration.

- [X] **Step 4: Add typed paths/getters to `DataService.load`**

Extend with keyword-only args:

```python
object_path: str | Path | None = None
context_path: str | Path | None = None
transcript_path: str | Path | None = None
```

Add:

```python
def get_object_evidence(self, frame_id: str) -> ObjectEvidence | None: ...
def get_frame_context(self, frame_id: str) -> FrameContext | None: ...
def get_transcript_segments(self, video_id: str, start_ms: int, end_ms: int) -> list[TranscriptSegment]: ...
```

Do not fuse modalities inside `DataService`.

- [X] **Step 5: Keep current text retrieval consumers working**

`get_evidence(frame_id, RetrievalSource.CAPTION/OCR/ASR)` must keep returning `str | None`. Context/object are not routed through `RetrievalSource` yet.

- [X] **Step 6: Run store/service tests**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/test_data_service_evidence.py -q
pyright src/hcmai/data/stores src/hcmai/data/pipeline.py
```

Expected: PASS.

- [X] **Step 7: Commit typed data access**

```bash
git add src/hcmai/data/stores src/hcmai/data/pipeline.py tests/data/test_data_service_evidence.py
git commit -m "refactor(data): expose typed enrichment stores"
```

---

### Task 9: Wire Configuration, Stage Commands, and End-to-End Enrichment Validation

**Files:**

- Modify: `configs/enrichment.yaml`
- Modify: `configs/preparation.s3.yaml:18-43,68-102`
- Modify: `src/hcmai/data/enrichment/pipeline.py`
- Modify: `scripts/generate_enrichment.py`
- Modify: `scripts/generate_ocr_enrichment.py`
- Modify: `scripts/generate_object_enrichment.py`
- Modify: `scripts/build_frame_context.py`
- Test: `tests/data/enrichment/test_enrichment_v1_integration.py`

**Interfaces:**

- Consumes: BTC canonical frame store, BTC object root, keyframes, videos for ASR.
- Produces: independently runnable artifacts with one cheap dependent context build; no index build in this task.

- [X] **Step 1: Write a tiny integration test with two frames and injected fake adapters**

The fixture must include:

- two canonical BTC frames;
- caption fake where frame 1 completes and frame 2 fails;
- OCR fake with one valid region for frame 1 and no text for frame 2;
- BTC object JSON with repeated `person` detections;
- no ASR dependency in context build.

Assert:

```python
context1 = contexts["f1"]
assert "[CAPTION]" in context1.context_text
assert "[VISIBLE_TEXT]" in context1.context_text
assert "person x2" in context1.context_text

context2 = contexts["f2"]
assert "None" not in (context2.context_text or "")
assert "[CAPTION]" not in (context2.context_text or "")
```

Then change only `FrameContextConfig.object_token_budget` and assert Caption/OCR/Object generation functions are not called while the context artifact rebuilds.

- [X] **Step 2: Run the integration test and verify the stage wiring is incomplete**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/data/enrichment/test_enrichment_v1_integration.py -q
```

Expected: FAIL until all paths/config are wired.

- [X] **Step 3: Expand `configs/enrichment.yaml` into the V1 artifact contract**

Use this structure:

```yaml
dataset:
  version: "hcmai2026_v1"
  source: "btc_keyframes"
  btc_root: "data"
  data_root: "data"
  frame_store_id: "btc-keyframes-v1"
  frames_path: "artifacts/frame_store/frames.parquet"
  frame_store_output: "artifacts/frame_store"
  objects_root: "data/objects"

caption:
  output_dir: "artifacts/enrichment/captions"
  # retain existing pinned model/decoding settings

ocr:
  output_dir: "artifacts/enrichment/ocr"
  artifact_version: "ocr-v1"
  min_region_confidence: 0.0
  min_context_quality: 0.5
  # retain existing pinned model settings

objects:
  output_dir: "artifacts/enrichment/objects"
  artifact_version: "object-v1"
  summary_min_confidence: 0.25
  max_summary_labels: 20

transcript:
  output_dir: "artifacts/enrichment/transcripts"
  # retain existing pinned ASR/diarization settings

context:
  output_dir: "artifacts/enrichment/context"
  context_version: "frame-context-v1"
  caption_token_budget: 80
  ocr_token_budget: 80
  object_token_budget: 40
  min_ocr_quality: 0.5
```

- [X] **Step 4: Remove preprocessing from the competition preparation stage graph**

In `configs/preparation.s3.yaml`, remove TransNet/EfficientGEBD/DINO settings
from the active HCMAI 2026 competition preparation path; active stages become:

```yaml
stages:
  frame_store: true        # BTC import, not video preprocessing
  caption: true
  ocr: true
  objects: true
  asr: true
  frame_context: true
  visual_index: false
  caption_index: false
  ocr_index: false
  asr_index: false
```

Index stages are deliberately false here because retrieval/index design is separate.

- [X] **Step 5: Keep all model-heavy stages independently runnable**

The public facade must support these independent commands without an all-or-nothing orchestration lock:

```text
prepare BTC frame store
caption
OCR
object import
ASR
frame context
```

`frame context` validates its three upstream artifacts and fails fast only on missing/corrupt artifact-level prerequisites; failed rows inside a valid specialist artifact are omitted per frame.

- [X] **Step 6: Run the integration test and focused enrichment suite**

```bash
PYTHONPATH=.:src aic/bin/python -m pytest \
  tests/data/test_evidence_contracts.py \
  tests/data/test_btc_ingestion.py \
  tests/data/enrichment/test_caption_evidence.py \
  tests/data/enrichment/test_ocr_evidence.py \
  tests/data/enrichment/test_object_evidence.py \
  tests/data/enrichment/test_asr_segment_evidence.py \
  tests/data/enrichment/test_frame_context.py \
  tests/data/test_data_service_evidence.py \
  tests/data/enrichment/test_enrichment_v1_integration.py -q
```

Expected: PASS with no remote inference/network calls.

- [X] **Step 7: Commit configuration and stage wiring**

```bash
git add configs src/hcmai/data/enrichment/pipeline.py scripts tests/data/enrichment/test_enrichment_v1_integration.py
git commit -m "feat(enrichment): wire HCMAI enrichment v1 stages"
```

---

### Task 10: Update Documentation and Run the Repository Gate

**Files:**

- Modify: `src/hcmai/data/README.md:1-110`
- Modify: `src/hcmai/data/enrichment/README.md:1-81`
- Modify: `scripts/README.md:59-154`
- Modify: `docs/superpowers/specs/2026-08-17-enrichment-pipeline-v1-design.md` only if implementation uncovered a factual contradiction; do not change frozen design choices.

**Interfaces:**

- Consumes: final implementation.
- Produces: operator instructions consistent with BTC-native enrichment and a clean release gate.

- [X] **Step 1: Replace preprocessing-first diagrams with BTC-native enrichment diagrams**

Document this exact high-level flow:

```text
BTC keyframes ──> Caption ──────┐
              └─> OCR ──────────┤
BTC objects ────> Object Import ├─> FrameContext V1
Videos ─────────> ASR segments  │   (ASR excluded)
                                 └─> specialist artifacts
```

Keep a separate note that only `src/hcmai/data/preprocessing/video.py` remains
as a low-level utility; custom frame extraction is not part of the competition
profile.

- [X] **Step 2: Document the source-of-truth artifacts and legacy projections**

The docs must clearly distinguish:

```text
captions.parquet                source of truth
ocr/frames.parquet             source of truth
ocr/regions.parquet            source of truth
objects/frames.parquet         source of truth
objects/detections.parquet     source of truth
transcripts/*.parquet          source of truth
context/frame_context_v1.parquet derived cross-modal view
frame_enrichment.parquet       temporary compatibility projection only
```

- [X] **Step 3: Document exact stage commands**

Include runnable commands for BTC ingest, Caption, OCR, Object import, ASR, and FrameContext build. Do not document an index build as part of this V1 enrichment plan.

- [X] **Step 4: Run whitespace/type/focused tests**

```bash
git diff --check
pyright src/hcmai/common/schemas src/hcmai/data/enrichment src/hcmai/data/stores src/hcmai/data/pipeline.py
PYTHONPATH=.:src aic/bin/python -m pytest tests/data tests/data/enrichment -q
```

Expected: no type errors in touched modules and all new data tests pass.

- [X] **Step 5: Run the repository release gate**

```bash
scripts/validate_repository.sh
```

Expected: the repository's existing deterministic backend/frontend gate passes; no test should invoke hosted inference or rebuild the real corpus.

- [X] **Step 6: Verify artifact dependency behavior manually with a tiny local run**

Run a two-frame fixture through Caption/OCR/Object/Context. Record file mtimes/hashes, change only `context.object_token_budget`, rerun Context, and verify:

```text
captions.parquet       unchanged
ocr/frames.parquet     unchanged
objects/frames.parquet unchanged
frame_context_v1.parquet changed
```

Then change OCR artifact version and verify Context rebuilds while Caption/Object artifacts remain untouched.

- [X] **Step 7: Commit docs and final gate**

```bash
git add src/hcmai/data/README.md src/hcmai/data/enrichment/README.md scripts/README.md
git commit -m "docs(data): document BTC-native enrichment pipeline"
```

---

## Implementation Order and Review Gates

Execute tasks in this order because later interfaces depend on earlier contracts:

```text
1 Contracts
   ↓
2 BTC ingestion
   ↓
3 Caption ─────┐
4 OCR ─────────┼──> 7 FrameContext
5 Objects ─────┘
6 ASR (parallel; no FrameContext dependency)
   ↓
8 Typed DataService
   ↓
9 V1 configuration/integration
   ↓
10 Docs + release gate
```

Tasks 3, 4, 5, and 6 are independent once Task 1 is merged and can be implemented in parallel worktrees/subagents. Task 7 waits for 3/4/5. Task 8 should land after the specialist artifact schemas are stable. Task 9 is the integration gate.

## Definition of Done

The implementation is complete only when all of the following are demonstrated by tests/artifacts:

- `DataService.prepare()` for the HCMAI 2026 profile imports BTC keyframes and does not call video preprocessing.
- Caption writes a versioned `CaptionEvidence` source-of-truth artifact and retries only failed/incomplete rows.
- OCR preserves raw region rows/bounding boxes, independently derives normalized text/quality, and never overwrites raw evidence.
- BTC object import preserves every valid repeated detection and bounding box; derived counts do not lose multiplicity.
- ASR remains timestamped segment evidence; its version is absent from `frame_context-v1` dependency identity.
- `frame_context-v1` contains only Caption + usable normalized OCR + Object summary, in the frozen section order, with deterministic 80/80/40 whitespace-token budgets.
- Missing/failed modality rows are omitted per frame without failing the corpus.
- Changing only context config rebuilds only context.
- Changing OCR invalidates OCR-derived views/context but does not rerun Caption, Objects, ASR, or Visual work.
- Existing retrieval code can continue temporarily through compatibility projections, while new typed stores expose clean inputs for the next Indexing/Retrieval plan.
- Full repository validation passes.

## Self-Review Results

- **Spec coverage:** All frozen V1 decisions, success criteria, failure/resume semantics, artifact ownership, version lineage, and current-code compatibility issues are mapped to Tasks 1–10.
- **Scope:** Retrieval/index fusion, query routing, sparse OCR indexing, object structured retrieval, scene fusion, and KIS/VQA/TRAKE behavior are intentionally excluded and must be handled by the next design/plan.
- **Type consistency:** The plan uses one set of public contract names (`CaptionEvidence`, `OCREvidence`, `OCRRegion`, `ObjectDetection`, `ObjectEvidence`, `FrameContext`) throughout.
- **Placeholder scan:** No implementation step depends on TBD/TODO behavior. V1 serializer budgets, OCR normalization, object summary ordering, lineage fields, and failure semantics are explicit.
