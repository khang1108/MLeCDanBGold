# Scripts Domain Layout Design

**Date:** 2026-09-10<br>
**Status:** Approved for implementation<br>
**Scope:** Repository-root `scripts/` only

## Goal

Replace the flat root `scripts/` layout with domain-oriented subpackages so executable ownership is visible from the path, while preserving each script's behavior and updating every active in-repository caller.

## Decisions

- Perform a clean move. Old paths such as `scripts/build_retrieval_indexes.py` are removed rather than retained as compatibility wrappers.
- Preserve filenames and command-line arguments; this change reorganizes ownership and does not redesign individual CLIs.
- Use importable Python packages and invoke Python CLIs with `python -m`.
- Keep shell automation executable by path.
- Keep `tests/scripts/` flat for now; update its imports and path constants without expanding the scope into a test-layout reorganization.
- Update active documentation and automation. Historical design specifications and implementation plans retain the paths that were true when they were written.
- Do not modify unrelated technical-report work. If an active report contains a moved path, update only that exact reference.

## Target Layout

```text
scripts/
├── __init__.py
├── README.md
├── corpus/
│   ├── __init__.py
│   ├── extract_custom_keyframes.py
│   ├── export_corpus_jsonl.py
│   ├── ingest_btc_keyframes.py
│   ├── materialize_custom_frames.py
│   └── prepare_custom_pipeline.py
├── enrichment/
│   ├── __init__.py
│   ├── build_frame_context.py
│   ├── detect_objects.py
│   ├── generate_enrichment.py
│   ├── generate_ocr_enrichment.py
│   ├── prepare_transcripts.py
│   └── translate.py
├── indexing/
│   ├── __init__.py
│   └── build_retrieval_indexes.py
├── evaluation/
│   ├── __init__.py
│   ├── benchmark_remote.py
│   ├── evaluate_benchmark.py
│   ├── evaluate_benchmark_v2.py
│   └── evaluate_temporal_p0.py
├── diagnostics/
│   ├── __init__.py
│   ├── debug_temporal_evidence.py
│   └── diagnose_temporal_quality.py
├── review/
│   ├── __init__.py
│   ├── build_overview.py
│   ├── build_review_sheets.py
│   └── manual_label_unlabeled.py
└── automation/
    └── watch_github_main.sh
```

## Ownership Boundaries

### `corpus`

Creates, materializes, orchestrates, or exports corpus records and frame stores. The custom pipeline belongs here because its unit of work is a custom corpus lifecycle, even though it coordinates downstream enrichment and index stages.

### `enrichment`

Produces specialist frame or transcript evidence: captions, OCR, objects, ASR transcripts, translated captions, and combined frame context.

### `indexing`

Builds and validates serving indexes. The current builder is large and operationally distinct, so it receives a dedicated domain even though it is presently the only file.

### `evaluation`

Runs benchmark requests or computes benchmark and experiment metrics from saved results.

### `diagnostics`

Inspects evidence components and retrieval/alignment quality without serving as a benchmark evaluator.

### `review`

Builds human-review artifacts or records manual labels.

### `automation`

Contains repository operational automation that is neither a Python data pipeline nor an experiment.

## Invocation Contract

Python scripts become module entry points and continue to require the repository root plus `src` on `PYTHONPATH`:

```bash
PYTHONPATH=.:src aic/bin/python -m scripts.corpus.ingest_btc_keyframes --help
PYTHONPATH=.:src aic/bin/python -m scripts.enrichment.detect_objects --help
PYTHONPATH=.:src aic/bin/python -m scripts.indexing.build_retrieval_indexes --help
PYTHONPATH=.:src aic/bin/python -m scripts.evaluation.evaluate_benchmark_v2 --help
PYTHONPATH=.:src aic/bin/python -m scripts.diagnostics.debug_temporal_evidence --help
```

Shell automation remains path-based:

```bash
scripts/automation/watch_github_main.sh --help
```

No root-level Python wrapper remains. A caller using an old path must migrate to its new module name.

## Internal Integration Changes

### Imports

Tests and internal callers migrate from flat imports such as:

```python
from scripts import prepare_custom_pipeline
from scripts.evaluate_benchmark_v2 import evaluate
```

to domain imports such as:

```python
from scripts.corpus import prepare_custom_pipeline
from scripts.evaluation.evaluate_benchmark_v2 import evaluate
```

Each Python directory receives an `__init__.py` package marker. Package initializers do not eagerly import CLIs or expose a public facade; callers import the owning module explicitly.

### Custom pipeline subprocesses

`prepare_custom_pipeline` currently joins `PROJECT_ROOT / "scripts" / <filename>` and launches files. It will instead launch explicit modules:

```python
subprocess.run(
    [sys.executable, "-m", module_name, *arguments],
    cwd=PROJECT_ROOT,
    check=True,
)
```

Stage identifiers become fully qualified module constants under `scripts.enrichment` and `scripts.indexing`. This removes sibling-path assumptions and keeps stage ownership visible in logs and tests.

### Repository-root discovery

Moved scripts that derive defaults from `Path(__file__)` must account for the added directory depth. Their resolved defaults must continue pointing to repository-level `configs/`, `artifacts/`, `data/`, and `runs/`; no default artifact location changes are allowed.

## Reference Migration

Update current references in:

- root `README.md` and `KNOWLEDGE.md`;
- `scripts/README.md`;
- active `offline/` documentation;
- active runbooks and shell workflows under `docs/runbooks/`;
- active component documentation under `src/`;
- tests that import modules or execute a script path;
- the exact technical-report path reference where required.

Do not rewrite archived documents under `docs/superpowers/plans/` or `docs/superpowers/specs/`; their old paths document historical repository states. Ignore generated bytecode under `__pycache__/` and remove no unrelated local files.

## Error Handling and Compatibility

- CLI parsing, exit codes, stdout/stderr behavior, and artifact schemas remain unchanged.
- Import or subprocess failures must expose the new fully qualified module name.
- The migration must not add aliases through `sys.modules`, symlinks, deprecated wrappers, or fallback path probing.
- A repository search after migration must find no old flat path in active code or documentation, excluding explicitly historical plans/specifications.

## Verification

1. Compile every moved Python module.
2. Run `--help` through `python -m` for every moved Python CLI.
3. Run `bash -n` and the existing tests for `watch_github_main.sh`.
4. Run script-focused tests plus affected corpus, enrichment, retrieval-index, and baseline evaluator tests.
5. Run Pyright against `scripts` and affected tests.
6. Run the full Python suite and report unrelated pre-existing or environment-dependent failures separately rather than hiding them.
7. Search active repository content for every removed flat script path.
8. Run `git diff --check`.
9. Inspect Git status to ensure pre-existing technical-report and baseline changes were preserved.

## Acceptance Criteria

- The root of `scripts/` contains no executable implementation other than grouped directories; only package/documentation support files remain.
- All 22 tracked executables appear in exactly one declared domain.
- Every active internal Python import resolves through the new package path.
- Every maintained command example and runbook uses the new entry point.
- `prepare_custom_pipeline` launches its stages by module name and preserves stage order and arguments.
- Moved scripts resolve the same repository-relative defaults as before.
- Focused tests, compilation, static analysis, CLI help checks, shell syntax checks, and whitespace validation pass.
- No compatibility wrapper or unrelated refactor is introduced.
