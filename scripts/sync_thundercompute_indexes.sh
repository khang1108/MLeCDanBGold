#!/usr/bin/env bash
# Transfer only fast-track indexing inputs or validated retrieval bundles.

set -euo pipefail

: "${HCMAI_THUNDER_HOST:?set HCMAI_THUNDER_HOST, e.g. ssh alias or user@host}"
: "${HCMAI_THUNDER_ROOT:?set HCMAI_THUNDER_ROOT to the repo/data root on ThunderCompute}"
: "${HCMAI_LOCAL_ROOT:?set HCMAI_LOCAL_ROOT to the local HCMAI working root}"

if [[ "${HCMAI_LOCAL_ROOT}" != /* ]]; then
    echo "HCMAI_LOCAL_ROOT must be an absolute path" >&2
    exit 2
fi
if [[ "${HCMAI_THUNDER_ROOT}" != /* ]]; then
    echo "HCMAI_THUNDER_ROOT must be an absolute path" >&2
    exit 2
fi
if [[ ! -d "${HCMAI_LOCAL_ROOT}" ]]; then
    echo "HCMAI_LOCAL_ROOT does not exist: ${HCMAI_LOCAL_ROOT}" >&2
    exit 2
fi

local_root="${HCMAI_LOCAL_ROOT%/}"
remote_root="${HCMAI_THUNDER_ROOT%/}"
remote_destination="${HCMAI_THUNDER_HOST}:${remote_root}/"

push_relative() {
    local relative_path="$1"
    rsync -a --protect-args --relative -- \
        "${local_root}/./${relative_path}" \
        "${remote_destination}"
}

pull_bundle() {
    local bundle_name="$1"
    mkdir -p "${local_root}/artifacts/indexes/${bundle_name}"
    rsync -a --protect-args -- \
        "${HCMAI_THUNDER_HOST}:${remote_root}/artifacts/indexes/${bundle_name}/" \
        "${local_root}/artifacts/indexes/${bundle_name}/"
}

push_inputs() {
    # Canonical identities and their commit marker.
    push_relative "artifacts/frame_store/frames.parquet"
    push_relative "artifacts/frame_store/manifest.json"

    # BTC keyframes and organizer mapping only; raw source videos are excluded.
    push_relative "data/keyframes/"
    push_relative "data/map_keyframes/"

    # Derived typed context and segment-native transcript evidence.
    push_relative "artifacts/enrichment/context/frame_context_v1.parquet"
    push_relative "artifacts/enrichment/context/manifest.json"
    push_relative "artifacts/enrichment/transcripts/"

    # Reproducible build configuration and the source needed by the CLI.
    push_relative "configs/indexing.yaml"
    push_relative "configs/indexing.models.yaml"
    push_relative "src/"
    push_relative "scripts/build_retrieval_indexes.py"
    push_relative "pyproject.toml"
}

pull_indexes() {
    pull_bundle "visual"
    pull_bundle "context"
    pull_bundle "asr_segments"
    mkdir -p "${local_root}/artifacts/indexes"
    rsync -a --protect-args -- \
        "${HCMAI_THUNDER_HOST}:${remote_root}/artifacts/indexes/build_report.json" \
        "${local_root}/artifacts/indexes/build_report.json"
}

case "${1:-}" in
    push-inputs)
        push_inputs
        ;;
    pull-indexes)
        pull_indexes
        ;;
    *)
        echo "usage: $0 {push-inputs|pull-indexes}" >&2
        exit 2
        ;;
esac
