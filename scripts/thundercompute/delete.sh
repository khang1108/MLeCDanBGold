#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly DEFAULT_STATE_FILE="${REPO_ROOT}/.thundercompute/instance-id"

INSTANCE_ID="${HCMAI_THUNDER_INSTANCE_ID:-}"
INSTANCE_ID_FILE="${HCMAI_THUNDER_INSTANCE_ID_FILE:-${DEFAULT_STATE_FILE}}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

load_token_from_file() {
    local token_file=${TNR_API_TOKEN_FILE:-}
    local line

    [[ -n "${token_file}" && -r "${token_file}" ]] || return 0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" || "${line}" == \#* ]] && continue
        export TNR_API_TOKEN="${line}"
        return 0
    done < "${token_file}"
}

usage() {
    cat <<'EOF'
Usage:
  scripts/thundercompute/delete.sh [--instance ID]

The instance ID is read from --instance, HCMAI_THUNDER_INSTANCE_ID, or
.thundercompute/instance-id.  TNR_API_TOKEN or TNR_API_TOKEN_FILE is required
for programmatic authentication.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --instance)
            (( $# >= 2 )) || die "--instance requires a value."
            INSTANCE_ID=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1. Use --help."
            ;;
    esac
done

if [[ -z "${INSTANCE_ID}" && -f "${INSTANCE_ID_FILE}" ]]; then
    IFS= read -r INSTANCE_ID < "${INSTANCE_ID_FILE}" || true
fi

[[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] \
    || die "No valid instance ID found. Pass --instance ID or set the state file."
command -v tnr >/dev/null 2>&1 || die "Required command not found: tnr"
load_token_from_file

printf 'Deleting Thunder instance %s.\n' "${INSTANCE_ID}"
tnr delete --yes "${INSTANCE_ID}" </dev/null

if [[ -e "${INSTANCE_ID_FILE}" ]]; then
    [[ ! -L "${INSTANCE_ID_FILE}" ]] \
        || die "Instance-ID state path must not be a symbolic link."
    : > "${INSTANCE_ID_FILE}"
fi
printf 'Thunder instance %s deleted.\n' "${INSTANCE_ID}"
