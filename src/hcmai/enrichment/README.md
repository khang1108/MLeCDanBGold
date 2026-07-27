# Frame enrichment

`hcmai.enrichment` contains offline, resumable features that add searchable
evidence to canonical frame records. Each feature owns its configuration,
models, backend boundary, artifact handling, and pipeline.

## Structure

```text
enrichment/
├── caption/
│   ├── config.py       # CaptionConfig and YAML-backed CaptionJobConfig
│   ├── backend.py      # Lazy Florence caption model
│   ├── artifacts.py    # Resume validation and atomic artifacts
│   ├── resume.py       # Prior-row and manifest compatibility
│   ├── runner.py       # Image loading and caption batches
│   ├── report.py       # Reproducibility manifest
│   ├── pipeline.py     # Caption orchestration and CLI
│   └── __main__.py     # python -m entry point
└── ocr/
    ├── config.py       # OCRConfig
    ├── models.py       # OCRResult and OCR-local row types
    ├── protocols.py    # OCRBackend boundary
    ├── backend.py      # Lazy Florence OCR model
    ├── artifacts.py    # Validation and artifact writes
    ├── report.py       # OCR run report
    └── pipeline.py     # Resume and batch orchestration
```

Caption and OCR types stay in their respective packages unless another
feature consumes the same contract. Cross-feature output is validated with
the shared `hcmai.common.schemas.FrameEnrichment` contract.

## Caption job

Caption settings are read from
[`configs/enrichment.yaml`](../../../configs/enrichment.yaml). Relative dataset
and output paths are resolved from the repository root.

```bash
PYTHONPATH=src aic/bin/python -m hcmai.enrichment.caption
```

The equivalent thin script is:

```bash
PYTHONPATH=src aic/bin/python scripts/generate_enrichment.py
```

Both accept `--config`, `--frames`, `--dataset-root`, and `--output`
overrides. The job writes:

```text
frame_enrichment.parquet
manifest.json
failures.json
```

Completed rows with a compatible manifest are skipped. Failed, incomplete,
or malformed rows are retried without changing canonical `frame_id`,
`video_id`, or `frame_idx` values.

## OCR job

`generate_ocr` accepts an `OCRConfig`, a frame Parquet path, an artifact
directory, and an optional injected `OCRBackend`. Model loading is lazy and
tests use fake backends; tests must not download a checkpoint.

OCR writes the same shared enrichment table format plus its own manifest,
failure list, and report. Join caption and OCR artifacts on `frame_id`.

## Dependency direction

```text
caption ─┐
         ├──> common.schemas.FrameEnrichment
ocr ─────┘
```

`common` must not import from `enrichment`. Feature-specific configs, backend
protocols, intermediate results, and report types remain inside the owning
feature.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_caption.py tests/test_ocr.py tests/test_config.py
```

Real corpus runs and generated artifacts are local evidence and must not be
committed.
