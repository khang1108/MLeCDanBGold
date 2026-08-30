# Phase B Task 1 report

## Files inspected

- `AGENTS.md`
- `.superpowers/sdd/2026-08-30-hcmai-phase-b-corpus-schema-cleanup/task-1-brief.md`
- `src/hcmai/common/config.py`
- `src/hcmai/common/schemas/frame.py`
- `src/hcmai/common/schemas/evidence.py`
- `src/hcmai/common/schemas/transcript.py`
- `src/hcmai/data/pipeline.py`
- `src/hcmai/data/stores/frame.py`
- `src/hcmai/data/stores/evidence.py`
- `src/hcmai/data/enrichment/transcripts/store.py`
- `tests/test_data_loader.py`
- `tests/data/test_data_service_evidence.py`

## Files changed

- `tests/compat/test_current_artifact_layout.py`
- `tests/compat/test_runtime_loaders.py`
- This report.

No production artifacts were copied and no files under `tests/fixtures/artifacts/`
were needed. All Parquet fixtures are generated in pytest `tmp_path` directories.

## Design choices

- Frozen the exact configured frame, enrichment, transcript, and index paths
  from the brief using `AppConfig()`.
- Used one canonical frame row to assert `frame_id`, `video_id`, `frame_idx`,
  `timestamp_ms`, `image_path`, and nullable `thumbnail_path`.
- Exercised current `DataService`/typed stores for caption, OCR, zero-detection
  object evidence, and segment-native transcripts.
- Explicitly asserted that absent caption/OCR text and optional transcript
  provenance remain `None`, rather than becoming empty or fabricated values.

## Verification

- `PYTHONPATH=src pytest tests/compat -v` — failed initially because the
  default interpreter has no Parquet engine (`pyarrow`/`fastparquet`).
- `PYTHONPATH=src /tmp/hcmai-compat-venv/bin/pytest tests/compat -v` — **3 passed**.

The temporary environment was used only because the default interpreter lacked
the repository's declared runtime dependencies.

## Commit

Commit hash: `PENDING`

## Concerns

- The default system interpreter cannot run Parquet-based compatibility tests
  without installing the declared dependencies. No repository behavior is
  affected.

