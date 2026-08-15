#!/usr/bin/env bash
set -euo pipefail

# Thunder batch launcher for S3 corpus preparation
#
# Usage:
#   scripts/thunder_batch_launcher.sh [--limit N] [--delete-instance]

LIMIT=""
DELETE_INSTANCE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --delete-instance)
      DELETE_INSTANCE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=== Thunder Batch Launcher ==="

# Check prerequisites
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU is required."
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "Disk Status:"
df -h /

# Credentials Injection
# Thunder provides secrets via environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN)
# We ensure we don't copy or persist plaintext credentials to disk.
if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "ERROR: AWS credentials not found in environment."
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not found in environment."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
# Ensure virtualenv exists, this script expects to be run in the repo root
if [[ ! -d "aic" ]]; then
    python3 -m venv aic
fi
source aic/bin/activate
pip install -e '.[s3,preprocessing,transcripts,embedding]'

# Run the preparation pipeline under persistent logging
LOG_FILE="runs/thunder_batch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p runs

CMD=("python" "scripts/prepare_s3_corpus.py")
if [[ -n "$LIMIT" ]]; then
    CMD+=("--limit" "$LIMIT")
fi

echo "Starting pipeline... Logging to $LOG_FILE"
set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Pipeline completed successfully."
else
    echo "WARNING: Pipeline failed with exit code $EXIT_CODE. Please review logs."
    echo "WARNING: Billing will continue until instance is manually stopped!"
    exit $EXIT_CODE
fi

if [[ $DELETE_INSTANCE -eq 1 ]]; then
    echo "Opt-in instance deletion requested. Verified upload successful."
    echo "Terminating instance..."
    sudo poweroff
fi
