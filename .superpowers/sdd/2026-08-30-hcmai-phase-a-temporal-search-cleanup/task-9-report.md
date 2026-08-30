# Task 9 — Reduce Temporal Search Configuration and Detach Reranker Wiring

## Result

Completed Phase A Task 9 against the frozen v2 cleanup spec on 2026-08-30.
`AlignmentConfig` now retains only the full-corpus temporal-baseline fields
(`lambda_gap`, `event_power`, `chunk_size`, `cluster_delta`) and rejects
retired shortlist keys instead of silently ignoring them. The checked-in
baseline YAML was cleaned to the same schema while preserving the existing
baseline values for the retained fields.

Default KIS and TRAKE runtime wiring remains visual retrieval plus temporal
search only. No default reranker is created, injected, or held by the loaded
service or either workflow head. The standalone reranking package remains in
place for explicit experiments.

## TDD record

### RED

Created `tests/orchestration/test_default_dependencies.py` and extended
`tests/config/test_search_config.py` with temporal-config coverage.

The task-brief command initially failed during collection for two environment
reasons unrelated to the cleanup logic:

1. `PYTHONPATH=src` does not expose the repository-root `thundercompute`
   package in this workspace, so config/setup tests require `PYTHONPATH=.:src`.
2. Importing `hcmai.orchestration.setup` pulls in retrieval modules that import
   `faiss`, which is not installed in this container.

The new dependency test now stubs `faiss` at import time so the wiring
contract can be checked without widening the runtime dependency surface.

### GREEN

Passed the task-focused suite with the corrected import path:

```text
PYTHONPATH=.:src pytest tests/config tests/orchestration/test_default_dependencies.py -v

6 passed in 0.72s
```

Also passed the touched legacy config regression:

```text
PYTHONPATH=.:src pytest tests/test_config.py -v

10 passed in 0.72s
```

### Review fix 1

The first version of `tests/orchestration/test_default_dependencies.py`
installed a fake `faiss` module at module import time when `faiss` was absent
from `sys.modules`. That approach could outlive the test itself and mask a real
installed package for later tests in the same session.

The test now:

- checks actual availability with `importlib.util.find_spec("faiss")`;
- only injects a fake `faiss` inside a `patch.dict(sys.modules, ...)` scope
  around the `hcmai.orchestration.setup` import path when FAISS is truly
  unavailable;
- unloads the imported `setup` and retrieval/index modules in `finally` so the
  temporary stub cannot leak into later tests.

Re-run after the review fix:

```text
PYTHONPATH=.:src pytest tests/config tests/orchestration/test_default_dependencies.py -v

6 passed in 0.64s
```

And the touched-file whitespace check:

```text
git diff --check -- src/hcmai/common/config.py src/hcmai/orchestration/setup.py \
  configs/baseline.yaml tests/config/test_search_config.py tests/test_config.py \
  tests/orchestration/test_default_dependencies.py src/hcmai/README.md \
  src/hcmai/common/schemas/README.md
```

## Runtime and config changes

- Removed `top_k`, `max_videos`, and `rrf_k` from
  `src/hcmai/common/config.py::AlignmentConfig`.
- Added `extra="forbid"` to `AlignmentConfig` so stale shortlist keys fail
  validation instead of being ignored.
- Removed the same retired keys from `configs/baseline.yaml` while preserving:
  `lambda_gap: 0.00001`, `event_power: 1.0`, `chunk_size: 65536`,
  `cluster_delta: 0.0`.
- Added a startup-composition regression test proving:
  `load_search_service()` keeps default required sources at `{visual}` and
  neither `service.kis` nor `service.trake` exposes a `reranking` dependency.
- Updated the baseline config regression in `tests/test_config.py` to match the
  reduced alignment schema.
- Clarified `src/hcmai/orchestration/setup.py` documentation so the default
  online baseline is described as visual-required with optional detached
  Context/segment-ASR retrieval and no reranker wiring.

## Natural in-scope doc cleanup

- Fixed `src/hcmai/README.md` to remove the stale `PipelineRegistry` reference
  from the runtime path and to describe the reduced `search.alignment` fields.
- Fixed `src/hcmai/common/schemas/README.md` so the request/response import
  example points at `hcmai.api.contracts` instead of the deleted
  `hcmai.common.schemas.search` module.

## Config sweep note

The required YAML grep after cleanup still reports:

- `configs/baseline.yaml` `alignment:`, `event_power`, and `cluster_delta`,
  which are expected retained baseline fields.
- `configs/prepare.yaml` `max_videos_per_batch`, which is an offline
  preparation setting outside `AlignmentConfig` and was left unchanged.

No checked-in runtime alignment YAML still contains the removed
`top_k`/`max_videos`/`rrf_k` keys.

## Files changed

- `configs/baseline.yaml`
- `src/hcmai/README.md`
- `src/hcmai/common/config.py`
- `src/hcmai/common/schemas/README.md`
- `src/hcmai/orchestration/setup.py`
- `tests/config/test_search_config.py`
- `tests/orchestration/test_default_dependencies.py`
- `tests/test_config.py`

## Research and compatibility

No scholarly research was needed for this task. The work implements the frozen
v2 spec and does not change retrieval scoring, temporal DP semantics, or model
selection. `KNOWLEDGE.md` was not changed.

This is intentionally breaking for retired alignment shortlist keys. No
compatibility shims were added. Unrelated dirty research/spec/test files in the
workspace were not staged.
