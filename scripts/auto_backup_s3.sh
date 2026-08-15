#!/usr/bin/env bash

# ============================================================
# HCMAI 2026 - Auto Backup Script for ThunderCompute
#
# Usage:
#   chmod +x scripts/auto_backup_s3.sh
#   ./scripts/auto_backup_s3.sh
# ============================================================

set -euo pipefail

# Configuration
SYNC_DIR="runs/"
S3_BACKUP_PATH="s3://mlecdanbgold-hcmai-hk/backup-runs/"
SYNC_INTERVAL_SECONDS=600  # 10 minutes

echo "============================================================"
echo "[HCMAI] Auto Backup to S3 Started"
echo "Directory: ${SYNC_DIR}"
echo "Destination: ${S3_BACKUP_PATH}"
echo "Interval: Every ${SYNC_INTERVAL_SECONDS} seconds"
echo "============================================================"

# Ensure AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "ERROR: aws-cli is not installed. Please install it first."
    exit 1
fi

while true; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Syncing ${SYNC_DIR} to S3..."
    
    # We use --quiet to avoid spamming the logs, but it will print errors if any
    aws s3 sync "${SYNC_DIR}" "${S3_BACKUP_PATH}" --quiet

    if [ $? -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Sync completed successfully. Sleeping for ${SYNC_INTERVAL_SECONDS}s..."
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: Sync encountered an error. Retrying in ${SYNC_INTERVAL_SECONDS}s..."
    fi

    sleep "${SYNC_INTERVAL_SECONDS}"
done
