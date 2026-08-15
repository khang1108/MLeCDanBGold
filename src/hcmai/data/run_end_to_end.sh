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
bash scripts/thunder_batch_launcher.sh

log "Step 2: Building Visual Embeddings & FAISS Index"
PYTHONPATH=src "${PYTHON_BIN}" scripts/build_embeddings.py \
  --config configs/baseline.yaml \
  --model-config llm/config.yaml \
  --dataset-root artifacts/frame_store \
  --frames artifacts/frame_store/frames.parquet \
  --output artifacts

log "Step 3: Generating Captions (Enrichment)"
PYTHONPATH=src "${PYTHON_BIN}" scripts/generate_enrichment.py

log "Step 4: Generating OCR Text (Enrichment)"
PYTHONPATH=src "${PYTHON_BIN}" scripts/generate_ocr_enrichment.py

log "Step 5: Generating ASR Transcripts (Enrichment)"
PYTHONPATH=src "${PYTHON_BIN}" scripts/prepare_transcripts.py

log "Step 6: Building Text Indexes (Caption, OCR, ASR)"
PYTHONPATH=src "${PYTHON_BIN}" scripts/build_caption_index.py --source caption
PYTHONPATH=src "${PYTHON_BIN}" scripts/build_caption_index.py --source ocr
PYTHONPATH=src "${PYTHON_BIN}" scripts/build_caption_index.py --source asr

log "Pipeline completed successfully! All artifacts and indexes are ready."
