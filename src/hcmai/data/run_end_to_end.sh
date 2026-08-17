#!/usr/bin/env bash
set -Eeuo pipefail

# HCMAI 2026 - End-to-End Data Pipeline Script
# Run this from the root of the repository.

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHON_BIN="aic/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Virtual environment python (${PYTHON_BIN}) not found!"
    echo "Please ensure you run this script from the project root and the 'aic' environment is set up."
    exit 1
fi

log() {
    printf '\n[Data Pipeline %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

log "Step 0: Setup TransNetV2 & EfficientGEBD models"
bash src/hcmai/data/setup_models.sh

log "Step 1: Starting Video Preprocessing (Extracting frames, GEBD, DINO dedup)"
"${PYTHON_BIN}" scripts/prepare_s3_corpus.py --config configs/preparation.s3.yaml "$@"

log "Pipeline completed successfully! All artifacts and indexes are ready."
