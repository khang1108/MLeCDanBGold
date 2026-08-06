# Frame enrichment

`hcmai.enrichment` owns offline, resumable caption and OCR generation for
canonical frames. External callers use `EnrichmentService`; each feature keeps
its contracts/entities in `models/` and framework-specific model code in
`adapters/`.

```text
enrichment/
├── pipeline.py                         # EnrichmentService public facade
├── caption/
│   ├── generator.py                    # Caption job orchestration
│   ├── models/contracts.py             # Caption adapter contract
│   ├── adapters/{transformers,remote}.py
│   └── artifacts/config/report/resume/runner.py
└── ocr/
    ├── generator.py                    # OCR job orchestration
    ├── models/{contracts,entities}.py  # OCR contract and results
    ├── adapters/florence.py
    └── artifacts/config/report.py
```

## Public service

```python
from hcmai.enrichment.pipeline import EnrichmentService

caption_report = EnrichmentService.generate_captions(
    frames_path,
    output_dir,
    caption_config,
    dataset_root=dataset_root,
)
ocr_report = EnrichmentService.generate_ocr(
    frames_path,
    output_dir,
    ocr_config,
    dataset_root=dataset_root,
)
```

Production scripts call this service boundary. Tests may inject fake caption or
OCR adapters; production callers do not import feature adapters or generators
directly.

Caption settings are read from
[`configs/enrichment.yaml`](../../../configs/enrichment.yaml). The existing
entry points remain:

```bash
PYTHONPATH=src aic/bin/python -m hcmai.enrichment.caption
PYTHONPATH=src aic/bin/python scripts/generate_enrichment.py
```

Jobs write `frame_enrichment.parquet`, `manifest.json`, and `failures.json`.
Compatible completed rows are skipped; failed, incomplete, or malformed rows
are retried. Caption and OCR artifacts join on the exact canonical `frame_id`
and never rewrite `video_id` or `frame_idx`.

Feature-local configuration, adapter contracts, intermediate entities, and
reports stay inside the owning feature. Cross-feature output uses the shared
`hcmai.common.schemas.FrameEnrichment` contract, and `common` never imports
from enrichment.

## Verification

```bash
PYTHONPATH=src aic/bin/pytest tests/test_caption.py tests/test_ocr.py tests/test_config.py
pyright src/hcmai/enrichment
```

Real-corpus artifacts are local experiment evidence and must not be committed.
