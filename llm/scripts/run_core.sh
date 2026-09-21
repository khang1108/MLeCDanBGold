#!/usr/bin/env bash
set -Eeuo pipefail

# Core API: embeddings + generic text generation.
# Point api.iamphuckhang.dev at this process (default localhost:8100).

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HCMAI_LLM_PROFILE=core
export HCMAI_LLM_CONFIG="${HCMAI_LLM_CONFIG:-llm/config.yaml}"
export PYTHONPATH="${PYTHONPATH:-.}:src"

exec "$PYTHON_BIN" -m uvicorn llm.server.api:app \
  --host "$HOST" --port "$PORT" --workers 1 --proxy-headers
