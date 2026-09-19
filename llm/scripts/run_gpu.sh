#!/usr/bin/env bash
set -Eeuo pipefail

# Single-A6000 worker. Run exactly one heavy task at a time:
#   HCMAI_GPU_TASK=caption|ocr|objects|asr llm/scripts/run_gpu.sh
# Point gpu.iamphuckhang.dev at this process (default localhost:8200).

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8200}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${HCMAI_GPU_TASK:=caption}"
export HCMAI_GPU_TASK
export HCMAI_LLM_PROFILE=gpu
export HCMAI_LLM_CONFIG="${HCMAI_LLM_CONFIG:-llm/config.yaml}"
export HCMAI_ENRICHMENT_CONFIG="${HCMAI_ENRICHMENT_CONFIG:-configs/vbs_prepare.yaml}"
export PYTHONPATH="${PYTHONPATH:-.}:src"

case "$HCMAI_GPU_TASK" in
  caption|ocr|objects|asr) ;;
  *) echo "HCMAI_GPU_TASK must be caption, ocr, objects, or asr" >&2; exit 2 ;;
esac

exec "$PYTHON_BIN" -m uvicorn llm.server.api:app \
  --host "$HOST" --port "$PORT" --workers 1 --proxy-headers
