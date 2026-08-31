# Task 12 Report: Artifact Compatibility and Phase B Closeout

## Status

Phase B closeout is implemented. Existing corpus and retrieval artifacts load
through the new runtime owners without an artifact-path migration. Mandatory
compatibility fixtures, real-corpus identity comparison, real application
readiness, offline command checks, and native C++ tests pass.

The complete Python suite has five known legacy custom-pipeline CLI failures
that predate Task 12 and are unrelated to Tasks 9–11. They are recorded below
and were not changed because the affected script/tests are user-owned and
outside this closeout.

## Closeout changes

- Rewrote `src/hcmai/README.md` around the current runtime call path and
  package owners.
- Moved the surviving legacy data guides to `offline/README.md` and
  `offline/WORKFLOW.md` and updated them for offline construction ownership.
- Updated the moved enrichment guide and runtime embedding/retrieval guides.
- Documented `Corpus.open(...)` as read-only access to existing artifacts,
  runtime retrieval as load/search only, and `offline.embeddings` /
  `offline.indexes` as construction owners.
- Retained the existing configured paths; no new artifact layout is
  documented or implemented.
- Removed the retired `src/hcmai/data/__init__.py`, its now-empty directories,
  210 generated `.pyc` files, and 48 generated `__pycache__` directories after
  inspecting their contents. No tracked bytecode existed.

`KNOWLEDGE.md` was not updated because Task 12 makes no research or algorithmic
decision. Its existing user modification was preserved unstaged.

## Artifact-path scan

Command:

```bash
rg -n "artifacts/corpora|corpus_v[0-9]|immutable corpus|canonical snapshot" \
  src offline configs
```

Result: no versioned corpus path or canonical-snapshot convention exists. The
only match is
`offline/ingestion/corpus_build/config.py:293`, whose validation message says
`immutable corpus revision`. `git blame` traces that line to commit `fd207f6b`
on 2026-08-15, before Phase B; it validates a production revision token and
does not introduce an artifact path.

## Fixture compatibility

Initial system-environment command:

```bash
PYTHONPATH=.:src pytest tests/compat -v
```

Result: **1 passed, 2 failed** because system Python has neither `pyarrow` nor
`fastparquet`; both failures occur while the fixture calls
`DataFrame.to_parquet`, before any runtime compatibility assertion.

Dependency-complete project environment:

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests/compat -v
```

Result: **3 passed**.

## Real-corpus smoke

Gate inputs were available:

- `artifacts/frame_store/frames.parquet`
- `artifacts/indexes/visual/`

The baseline config and existing artifact paths were loaded through the current
composition functions, then passed through the real FastAPI lifespan and
`/health` endpoint. Optional remote inference was omitted from this offline
artifact smoke; no artifact path or runtime retrieval configuration was
overridden.

Result:

```text
frame_store_loaded = true
retriever_loaded = true
ready = true
total_frames = 177321
capabilities.search = true
retrieval_modalities.visual.active = true
startup_messages = []
```

The first two `TestClient` attempts were bounded after more than three minutes
without a response. A direct startup diagnostic showed corpus loading in
23.648 seconds and retrieval loading in 47.609 seconds. The definitive smoke
used the same service/app lifespan and called the registered health endpoint
directly, avoiding the environment's deprecated TestClient bridge.

## Deterministic identity comparison

Sample rule: first, middle, and last canonical frame rows for deterministic
video `L21_V001`. Raw existing Parquet rows were the pre-Phase-B expected
authority; current `Corpus` values were the actual values. Transcript evidence
used a fixed half-open window from `max(0, timestamp_ms - 2000)` through
`timestamp_ms + 2001`.

All expected and actual fields matched exactly:

### `L21_V001_keyframe_000001`

```text
video_id: L21_V001
frame_idx: 0
timestamp_ms: 0
resolved image: /home/phuckhang/MyWorkspace/HCMAI_2026/data/keyframes/L21_V001/001.jpg
caption: An advertisement for a tv show with a city in the background.<pad>...
OCR: HTVO / 06:30:11 / giay
object labels: lantern, poster, skyscraper, tower
transcript segments: none
```

### `L21_V001_keyframe_000154`

```text
video_id: L21_V001
frame_idx: 18486
timestamp_ms: 616200
resolved image: /home/phuckhang/MyWorkspace/HCMAI_2026/data/keyframes/L21_V001/154.jpg
caption: A man sitting in front of a pile of books.<pad>...
OCR: HTV9 / 06:42:10 HD / Dy bao diem chuan phuang thuc xi cei qua thi nghiep trung hoc pho thong tang
object labels: clothing, human face, man, person
transcript segments: L21_V001_segment_000050, L21_V001_segment_000051
```

### `L21_V001_keyframe_000307`

```text
video_id: L21_V001
frame_idx: 37716
timestamp_ms: 1257230
resolved image: /home/phuckhang/MyWorkspace/HCMAI_2026/data/keyframes/L21_V001/307.jpg
caption: A red screen with the words Trung Tam Tin Tuc on it.<pad>...
OCR: HTVO / 07:00:21 HD / TRUNG TAM TIN TÜC
object labels: computer keyboard, musical instrument
transcript segments: none
```

Result: **PASS**. No difference, including a runtime-type-only difference, was
observed in the compared values.

## Offline command and output compatibility

Non-mutating `--help` checks passed for:

- `scripts/generate_enrichment.py`
- `scripts/generate_ocr_enrichment.py`
- `scripts/prepare_transcripts.py`
- `scripts/detect_objects.py`
- `scripts/build_retrieval_indexes.py`
- `scripts/ingest_btc_keyframes.py`
- `scripts/extract_custom_keyframes.py`

Seven fixture-generation/output tests, all writing only below pytest temporary
directories, passed for Caption, OCR, transcripts, objects, visual embeddings,
dense-index bundle filenames, and BTC keyframe ingestion:

```text
7 passed in 1.49s
```

## Python validation

```bash
PYTHONPATH=.:src aic/bin/python -m compileall -q src/hcmai offline
```

Result: **PASS**.

```bash
PYTHONPATH=.:src aic/bin/python -m pytest tests -v
```

Result: **862 passed, 5 failed in 28.54s**.

The five failures are:

- `tests/scripts/test_custom_pipeline.py::test_pipeline_downloads_and_reuses_default_media_info_layout`
- `tests/scripts/test_custom_pipeline.py::test_pipeline_rejects_media_info_zip_path_traversal`
- `tests/scripts/test_custom_pipeline.py::test_pipeline_forwards_yt_dlp_authentication_options`
- `tests/scripts/test_custom_pipeline.py::test_pipeline_surfaces_and_persists_native_extraction_failure`
- `tests/scripts/test_custom_pipeline.py::test_custom_pipeline_coordinates_every_local_artifact_stage`

Each test passes retired flat CLI arguments to
`scripts/prepare_custom_pipeline.py`; the active parser requires one of
`preflight`, `process-archive`, `status`, or `finalize`. Task 10 already
recorded these five failures as out of scope. Tasks 9–11 did not change this
CLI, so Task 12 does not rewrite the user-owned command or tests.

## Native C++ validation

```bash
cmake -S offline/keyframes/keyframes_extraction -B build/keyframes-extraction
cmake --build build/keyframes-extraction -j
ctest --test-dir build/keyframes-extraction --output-on-failure
```

Result: configure/build **PASS**; **10/10 tests passed**.

## Final architecture scans

```bash
rg -n "DataService|hcmai\.data|hcmai\.common\.schemas" src/hcmai offline tests
rg -n "from offline|import offline" src/hcmai
rg -n "hcmai\.orchestration" offline
```

Results:

- No live `DataService`, `hcmai.data`, or `hcmai.common.schemas` import remains.
  The two `hcmai.common.schemas` matches are package-absence assertions in
  tests.
- No `src/hcmai` runtime import of `offline` remains; offline construction
  owns its artifact readers independently of the runtime corpus.
- No offline dependency on `hcmai.orchestration` remains.
- `src/hcmai/data` no longer exists.
- No `.pyc` or `__pycache__` remains under `src` or `offline` after cleanup.
- `git diff --check` passes.

## Skips and concerns

- Mandatory fixture compatibility: **not skipped**; passed in the documented
  project environment.
- Real-corpus smoke: **not skipped**; artifacts were available and readiness
  passed.
- Offline checks: **not skipped**.
- C++ tests: **not skipped**.
- Concern: the repository-wide Python suite is not completely green because
  of the five known unrelated CLI expectation failures above. No Phase B
  compatibility, architecture, identity, or native test failed.
