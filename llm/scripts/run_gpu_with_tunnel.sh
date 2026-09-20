#!/usr/bin/env bash
set -Eeuo pipefail

# Run the HCMAI GPU API and Cloudflare Tunnel on the SAME Slurm GPU node.
#
# Examples:
#   HCMAI_GPU_TASK=caption \
#   CLOUDFLARED_TUNNEL=hcmai-gpu \
#   ./run_gpu_with_tunnel.sh
#
# Or, for a remotely-managed tunnel:
#   CLOUDFLARED_TOKEN='...' ./run_gpu_with_tunnel.sh
#
# Assumes this script is already running inside the allocated GPU node
# (e.g. after srun/salloc).

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8200}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/media02/lttung10/bin/cloudflared}"
CLOUDFLARED_TUNNEL="${CLOUDFLARED_TUNNEL:-}"
CLOUDFLARED_TOKEN="${CLOUDFLARED_TOKEN:-eyJhIjoiZDY1ZDUwM2E1NWM3YWFhODZjYjI3OWU1NzEzMTkyOWMiLCJ0IjoiMjIxZjNlZWItMGExNi00ZTk5LTljYTQtYWYxYjgyZjlmYmUwIiwicyI6Ik5tSTROalU0T1dNdE9UazRaUzAwTUdNd0xXRTRZbVV0TW1ZMk1EbGpOMk5rTmpNMiJ9}"
READY_URL="${READY_URL:-http://127.0.0.1:${PORT}/ready}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"

: "${HCMAI_GPU_TASK:=caption}"
export HCMAI_GPU_TASK
export HCMAI_LLM_PROFILE=gpu
export HCMAI_LLM_CONFIG="${HCMAI_LLM_CONFIG:-llm/config.yaml}"
export HCMAI_ENRICHMENT_CONFIG="${HCMAI_ENRICHMENT_CONFIG:-configs/vbs_prepare.yaml}"
export PYTHONPATH="${PYTHONPATH:-.}:src"

case "$HCMAI_GPU_TASK" in
  caption|ocr|objects|asr|all|enrichment) ;;
  *)
    echo "HCMAI_GPU_TASK must be caption, ocr, objects, asr, all, or enrichment" >&2
    exit 2
    ;;
esac

if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "cloudflared not found or not executable: $CLOUDFLARED_BIN" >&2
  exit 2
fi

if [[ -z "$CLOUDFLARED_TOKEN" && -z "$CLOUDFLARED_TUNNEL" ]]; then
  echo "Set either CLOUDFLARED_TOKEN or CLOUDFLARED_TUNNEL." >&2
  exit 2
fi

API_PID=""
TUNNEL_PID=""

cleanup() {
  trap - EXIT INT TERM

  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi

  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi

  wait "$TUNNEL_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[run_gpu] node: $(hostname)"
echo "[run_gpu] starting API on ${HOST}:${PORT}"

"$PYTHON_BIN" -m uvicorn llm.server.api:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --proxy-headers &
API_PID=$!

echo "[run_gpu] waiting for ${READY_URL}"

deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS "$READY_URL" >/dev/null 2>&1; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[run_gpu] API exited before becoming ready." >&2
    wait "$API_PID"
    exit $?
  fi

  if (( SECONDS >= deadline )); then
    echo "[run_gpu] API did not become ready within ${READY_TIMEOUT}s." >&2
    exit 1
  fi

  sleep 1
done

echo "[run_gpu] API is ready."
echo "[run_gpu] starting cloudflared"

if [[ -n "$CLOUDFLARED_TOKEN" ]]; then
  "$CLOUDFLARED_BIN" tunnel run --token "$CLOUDFLARED_TOKEN" &
else
  "$CLOUDFLARED_BIN" tunnel run "$CLOUDFLARED_TUNNEL" &
fi
TUNNEL_PID=$!

# Exit the wrapper as soon as either child dies.
while true; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[run_gpu] API stopped." >&2
    wait "$API_PID"
    exit $?
  fi

  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[run_gpu] cloudflared stopped." >&2
    wait "$TUNNEL_PID"
    exit $?
  fi

  sleep 2
done
