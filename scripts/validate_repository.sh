#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

if [[ -n "${HCMAI_PYTHON:-}" ]]; then
    PYTHON_BIN="${HCMAI_PYTHON}"
elif [[ -x "${REPOSITORY_ROOT}/aic/bin/python" ]]; then
    PYTHON_BIN="${REPOSITORY_ROOT}/aic/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" -m pytest -q \
    tests/unit/temporal \
    tests/unit/vqa \
    tests/unit/llm/test_multiframe_vqa.py \
    tests/integration/test_progressive_temporal_core.py \
    tests/integration/test_vqa_api.py \
    tests/integration/test_vqa_pipeline.py \
    tests/test_trake_align.py \
    tests/test_trake_submission.py \
    tests/integration/test_trake_api.py

"${PYTHON_BIN}" -m pytest -q

npm --prefix frontend test -- --watchAll=false --runInBand
npm --prefix frontend run build

git diff --check HEAD -- \
    .gitignore \
    src/hcmai/data/enrichment/caption/config.py \
    tests \
    scripts/validate_repository.sh \
    scripts/README.md
