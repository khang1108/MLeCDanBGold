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
python_command="${HCMAI_PYTHON:-python}"
bundle_names=("visual" "context" "asr_segments")
promotion_items=("visual" "context" "asr_segments" "build_report.json")
transfer_staging=""
promotion_backup=""
promotion_complete=0
declare -a backed_up_items=()
declare -a promoted_items=()

safe_remove_temp() {
    local directory="$1"
    if [[ -z "${directory}" || ! -e "${directory}" ]]; then
        return 0
    fi
    case "${directory}" in
        "${local_root}/artifacts/.indexes-transfer."*|"${local_root}/artifacts/.indexes-backup."*)
            rm -rf -- "${directory}"
            ;;
        *)
            echo "refusing to remove unexpected temporary path: ${directory}" >&2
            return 1
            ;;
    esac
}

rollback_promotion() {
    local item
    local live_root="${local_root}/artifacts/indexes"
    local index
    local rollback_failed=0

    for ((index=${#promoted_items[@]} - 1; index >= 0; index--)); do
        item="${promoted_items[index]}"
        if ! rm -rf -- "${live_root}/${item}"; then
            rollback_failed=1
        fi
    done
    for ((index=${#backed_up_items[@]} - 1; index >= 0; index--)); do
        item="${backed_up_items[index]}"
        if [[ -e "${promotion_backup}/${item}" ]]; then
            if [[ -e "${live_root}/${item}" || -L "${live_root}/${item}" ]] || \
                ! mv -- "${promotion_backup}/${item}" "${live_root}/${item}"; then
                rollback_failed=1
            fi
        fi
    done
    if [[ "${rollback_failed}" -ne 0 ]]; then
        echo "index promotion rollback needs manual recovery from ${promotion_backup}" >&2
        return 1
    fi
    safe_remove_temp "${promotion_backup}"
    promotion_backup=""
    backed_up_items=()
    promoted_items=()
}

cleanup_pull() {
    local status="$?"
    set +e
    if [[ "${promotion_complete}" -eq 0 && -n "${promotion_backup}" ]]; then
        rollback_promotion
    fi
    safe_remove_temp "${transfer_staging}"
    if [[ "${promotion_complete}" -eq 1 && -n "${promotion_backup}" ]]; then
        safe_remove_temp "${promotion_backup}"
    fi
    return "${status}"
}

trap cleanup_pull EXIT
trap 'exit 130' HUP INT TERM

push_relative() {
    local relative_path="$1"
    rsync -a --protect-args --relative -- \
        "${local_root}/./${relative_path}" \
        "${remote_destination}"
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

validate_staged_report() {
    local report_path="$1"
    if ! command -v "${python_command}" >/dev/null 2>&1; then
        echo "HCMAI_PYTHON is not executable: ${python_command}" >&2
        return 1
    fi
    "${python_command}" - "${report_path}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except Exception as error:
    raise SystemExit(f"invalid remote build report {path}: {error}") from error
if not isinstance(report, dict) or report.get("status") != "passed":
    raise SystemExit("remote build report status is not 'passed'")
indexes = report.get("indexes")
if not isinstance(indexes, dict):
    raise SystemExit("remote build report is missing indexes")
required = ("visual", "context", "asr_segments")
missing = [name for name in required if not isinstance(indexes.get(name), dict)]
if missing:
    raise SystemExit("remote build report is missing index entries: " + ", ".join(missing))
PY
}

validate_staged_bundles() {
    local staging_root="$1"
    local report_path="$2"
    local local_pythonpath="${local_root}/src"
    if [[ -n "${PYTHONPATH:-}" ]]; then
        local_pythonpath="${local_pythonpath}:${PYTHONPATH}"
    fi
    PYTHONPATH="${local_pythonpath}" "${python_command}" - \
        "${staging_root}" "${report_path}" <<'PY'
import json
from pathlib import Path
import sys

from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

staging_root = Path(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
dataset_version = report.get("dataset_version")
if not isinstance(dataset_version, str) or not dataset_version:
    raise SystemExit("remote build report has an invalid dataset_version")
reported_indexes = report["indexes"]
contracts = {
    "visual": (DenseIndex, "frame", "visual"),
    "context": (DenseIndex, "frame", "context"),
    "asr_segments": (SegmentDenseIndex, "segment", "asr"),
}

for name, (loader, entity_kind, retrieval_source) in contracts.items():
    bundle = staging_root / name
    # Loading is the authoritative v2 checksum/layout gate. Missing local
    # FAISS/Parquet dependencies fail this command closed before promotion.
    index = loader.load(bundle)
    metadata = index.metadata
    if metadata.schema_version != "dense-index-v2":
        raise SystemExit(f"staged {name} is not a dense-index-v2 bundle")
    if not isinstance(metadata.checksums, dict) or not metadata.checksums:
        raise SystemExit(f"staged {name} has no checksum manifest")
    for field in ("source_fingerprint", "config_fingerprint"):
        value = getattr(metadata, field)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"staged {name} has invalid {field}")
    expected = reported_indexes[name]
    actual_size = sum(
        path.stat().st_size for path in bundle.rglob("*") if path.is_file()
    )
    comparisons = {
        "size_bytes": actual_size,
        "vector_count": metadata.vector_count,
        "model_name": metadata.model_name,
        "model_revision": metadata.model_revision,
        "embedding_dim": metadata.embedding_dim,
        "normalization": metadata.normalization,
        "schema_version": metadata.schema_version,
        "entity_kind": metadata.entity_kind,
        "retrieval_source": metadata.retrieval_source,
        "source_fingerprint": metadata.source_fingerprint,
        "config_fingerprint": metadata.config_fingerprint,
        "checksums": metadata.checksums,
    }
    missing = [field for field in comparisons if field not in expected]
    if missing:
        raise SystemExit(
            f"remote build report {name} entry is missing: {', '.join(missing)}"
        )
    mismatched = [
        field for field, actual in comparisons.items() if expected[field] != actual
    ]
    if mismatched:
        raise SystemExit(
            f"staged {name} bundle disagrees with report: {', '.join(mismatched)}"
        )
    if metadata.dataset_version != dataset_version:
        raise SystemExit(f"staged {name} dataset_version disagrees with report")
    if metadata.entity_kind != entity_kind:
        raise SystemExit(f"staged {name} has unexpected entity_kind")
    if metadata.retrieval_source != retrieval_source:
        raise SystemExit(f"staged {name} has unexpected retrieval_source")
PY
}

promote_staged_indexes() {
    local live_root="${local_root}/artifacts/indexes"
    local item

    mkdir -p "${live_root}"
    promotion_backup="$(mktemp -d "${local_root}/artifacts/.indexes-backup.XXXXXX")"
    backed_up_items=()
    promoted_items=()

    for item in "${promotion_items[@]}"; do
        if [[ -e "${live_root}/${item}" || -L "${live_root}/${item}" ]]; then
            # Record intent before the atomic rename so an interrupt between
            # the rename and the next shell command can still restore it.
            backed_up_items+=("${item}")
            if ! mv -- "${live_root}/${item}" "${promotion_backup}/${item}"; then
                rollback_promotion
                return 1
            fi
        fi
    done

    for item in "${promotion_items[@]}"; do
        if [[ ! -e "${transfer_staging}/${item}" ]]; then
            echo "staged index transfer is missing ${item}" >&2
            rollback_promotion
            return 1
        fi
        # Every previous live item is already in the rollback directory.
        # Pre-record this target so signal-driven cleanup cannot miss a rename.
        promoted_items+=("${item}")
        if ! mv -- "${transfer_staging}/${item}" "${live_root}/${item}"; then
            rollback_promotion
            return 1
        fi
    done

    promotion_complete=1
    safe_remove_temp "${promotion_backup}"
    promotion_backup=""
    backed_up_items=()
    promoted_items=()
}

pull_indexes() {
    local bundle_name

    mkdir -p "${local_root}/artifacts"
    transfer_staging="$(mktemp -d "${local_root}/artifacts/.indexes-transfer.XXXXXX")"

    # The report is the remote publication gate and must pass before any large
    # bundle transfer starts. It is never written directly into the live root.
    rsync -a --protect-args -- \
        "${HCMAI_THUNDER_HOST}:${remote_root}/artifacts/indexes/build_report.json" \
        "${transfer_staging}/build_report.json"
    validate_staged_report "${transfer_staging}/build_report.json"

    for bundle_name in "${bundle_names[@]}"; do
        mkdir -p "${transfer_staging}/${bundle_name}"
        rsync -a --protect-args -- \
            "${HCMAI_THUNDER_HOST}:${remote_root}/artifacts/indexes/${bundle_name}/" \
            "${transfer_staging}/${bundle_name}/"
        if [[ -z "$(find "${transfer_staging}/${bundle_name}" -mindepth 1 -print -quit)" ]]; then
            echo "downloaded ${bundle_name} bundle is empty" >&2
            return 1
        fi
    done

    validate_staged_bundles \
        "${transfer_staging}" \
        "${transfer_staging}/build_report.json"
    promote_staged_indexes
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
