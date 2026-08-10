#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DEPLOY_SCRIPT="${SCRIPT_DIR}/deploy_cloudflared_private.sh"
readonly REMOTE_SCRIPT="/tmp/deploy_cloudflared_private.sh"
readonly REMOTE_LOG="/tmp/hcmai-deploy.log"

GPU=""
TOKEN=""
VCPUS=8
DISK_GB=100
WAIT_TIMEOUT_SECONDS=900
POLL_INTERVAL_SECONDS=10
TMUX_SESSION="hcmai-deploy"
INSTANCE_ID=""
DEPLOY_ARGS=()

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    if [[ -n "${INSTANCE_ID}" ]]; then
        printf 'ERROR: Thunder instance %s was created and may still be billable.\n' \
            "${INSTANCE_ID}" >&2
    fi
    exit 1
}

on_error() {
    local exit_code=$?
    trap - ERR
    printf 'ERROR: command failed at line %s (exit %s).\n' \
        "${BASH_LINENO[0]}" "${exit_code}" >&2
    if [[ -n "${INSTANCE_ID}" ]]; then
        printf 'ERROR: Thunder instance %s was created and may still be billable.\n' \
            "${INSTANCE_ID}" >&2
    fi
    exit "${exit_code}"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Usage:
  llm/launch_thunder_instance.sh --gpu a6000|l40 [options] \
      -- [deploy_cloudflared_private.sh options]

Required:
  --gpu GPU               GPU type: a6000 or l40.

Authentication:
  --token TOKEN           Thunder API token used only when the current tnr
                          login is invalid. TNR_API_TOKEN is also accepted.

Instance options:
  --vcpus COUNT           Virtual CPUs (default: 8).
  --disk GB               Primary disk size in GB (default: 100).
  --wait-timeout SECONDS  Maximum wait for RUNNING (default: 900).
  --poll-interval SECONDS Status polling interval (default: 10).
  --tmux-session NAME     Remote tmux session (default: hcmai-deploy).
  --help                  Show this help.

Bootstrap model options (place after --; all default to false):
  --caption true|false            Florence-2 frame captioning.
  --visual-embedding true|false   SigLIP2 visual/text embedding service.
  --caption-embedding true|false  BGE-M3 caption/text embedding service.
  --reranker true|false           Qwen3-VL visual reranking service.
  --vqa true|false                Grounded VQA model service.
  --conversation true|false       Alias for --vqa.
  --ocr true|false                OCR model service.

At least one bootstrap model must be set to true. Everything after -- is
forwarded unchanged to deploy_cloudflared_private.sh on the instance.

Examples:
  # CaptionStore generation and embedding
  llm/launch_thunder_instance.sh --gpu l40 --token "$TOKEN" -- \
      --caption true --caption-embedding true

  # Grounded VQA hosting
  llm/launch_thunder_instance.sh --gpu a6000 --token "$TOKEN" -- \
      --vqa true
EOF
}

require_value() {
    local option=$1
    local count=$2
    (( count >= 2 )) || die "${option} requires a value."
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            --gpu)
                require_value "$1" "$#"
                GPU="${2,,}"
                shift 2
                ;;
            --token)
                require_value "$1" "$#"
                TOKEN=$2
                shift 2
                ;;
            --vcpus)
                require_value "$1" "$#"
                VCPUS=$2
                shift 2
                ;;
            --disk)
                require_value "$1" "$#"
                DISK_GB=$2
                shift 2
                ;;
            --wait-timeout)
                require_value "$1" "$#"
                WAIT_TIMEOUT_SECONDS=$2
                shift 2
                ;;
            --poll-interval)
                require_value "$1" "$#"
                POLL_INTERVAL_SECONDS=$2
                shift 2
                ;;
            --tmux-session)
                require_value "$1" "$#"
                TMUX_SESSION=$2
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            --)
                shift
                DEPLOY_ARGS=("$@")
                return
                ;;
            *)
                die "Unknown option: $1. Use --help."
                ;;
        esac
    done
}

validate() {
    [[ "${GPU}" == "a6000" || "${GPU}" == "l40" ]] \
        || die "--gpu must be either a6000 or l40."
    (( ${#DEPLOY_ARGS[@]} > 0 )) \
        || die "Provide at least one bootstrap model option after --."
    is_positive_integer "${VCPUS}" || die "--vcpus must be a positive integer."
    is_positive_integer "${DISK_GB}" || die "--disk must be a positive integer."
    is_positive_integer "${WAIT_TIMEOUT_SECONDS}" \
        || die "--wait-timeout must be a positive integer."
    is_positive_integer "${POLL_INTERVAL_SECONDS}" \
        || die "--poll-interval must be a positive integer."
    [[ "${TMUX_SESSION}" =~ ^[A-Za-z0-9_-]+$ ]] \
        || die "--tmux-session may contain only letters, digits, underscores, and hyphens."
    [[ -f "${DEPLOY_SCRIPT}" && -r "${DEPLOY_SCRIPT}" ]] \
        || die "Private bootstrap not found or unreadable: ${DEPLOY_SCRIPT}"

    local command_name
    for command_name in tnr ssh python3; do
        command -v "${command_name}" >/dev/null 2>&1 \
            || die "Required command not found: ${command_name}"
    done
}

check_authentication() {
    if tnr status --no-wait --json >/dev/null 2>&1; then
        log "Thunder CLI is already authenticated."
        return
    fi

    if [[ -z "${TOKEN}" ]]; then
        TOKEN="${TNR_API_TOKEN:-}"
    fi
    [[ -n "${TOKEN}" ]] \
        || die "Thunder CLI is not authenticated; provide --token or TNR_API_TOKEN."

    log "Current Thunder login is unavailable; validating the supplied API token."
    export TNR_API_TOKEN="${TOKEN}"
    TOKEN=""
    tnr status --no-wait --json >/dev/null 2>&1 \
        || die "Thunder authentication failed with the supplied token."
    log "Thunder API token accepted for this process."
}

json_instance_id() {
    python3 -c '
import json
import sys

def find_identifier(value):
    if isinstance(value, dict):
        for key in ("identifier", "instance_id", "id"):
            candidate = value.get(key)
            if isinstance(candidate, (int, str)) and str(candidate).strip():
                return str(candidate)
        for candidate in value.values():
            found = find_identifier(candidate)
            if found is not None:
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = find_identifier(candidate)
            if found is not None:
                return found
    return None

identifier = find_identifier(json.load(sys.stdin))
if identifier is None:
    raise SystemExit("create response did not contain an instance identifier")
print(identifier)
'
}

json_instance_status() {
    local expected_id=$1
    python3 -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)
instances = payload if isinstance(payload, list) else [payload]

def values(item):
    if not isinstance(item, dict):
        return None, None
    identifier = item.get("identifier", item.get("instance_id", item.get("id")))
    status = item.get("status", item.get("state"))
    return identifier, status

for instance in instances:
    pending = [instance]
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, dict):
            identifier, status = values(candidate)
            if identifier is not None and str(identifier) == expected and status is not None:
                print(str(status).upper())
                raise SystemExit(0)
            pending.extend(candidate.values())
        elif isinstance(candidate, list):
            pending.extend(candidate)
raise SystemExit(1)
' "${expected_id}"
}

create_instance() {
    local create_response
    log "Creating Thunder instance: gpu=${GPU}, vcpus=${VCPUS}, disk=${DISK_GB}GB."
    create_response="$(tnr create \
        --gpu "${GPU}" \
        --num-gpus 1 \
        --vcpus "${VCPUS}" \
        --template base \
        --disk "${DISK_GB}" \
        --json)"
    INSTANCE_ID="$(json_instance_id <<<"${create_response}")"
    [[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] \
        || die "Thunder returned an invalid instance ID: ${INSTANCE_ID}"
    if [[ -n "${HCMAI_THUNDER_INSTANCE_ID_FILE:-}" ]]; then
        [[ -f "${HCMAI_THUNDER_INSTANCE_ID_FILE}" \
            && ! -L "${HCMAI_THUNDER_INSTANCE_ID_FILE}" ]] \
            || die "Instance-ID destination must be an existing regular file."
        printf '%s\n' "${INSTANCE_ID}" \
            > "${HCMAI_THUNDER_INSTANCE_ID_FILE}"
    fi
    log "Created Thunder instance ${INSTANCE_ID}."
}

wait_until_running() {
    local deadline status_response status
    deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
    log "Waiting up to ${WAIT_TIMEOUT_SECONDS}s for instance ${INSTANCE_ID} to run."

    while (( SECONDS < deadline )); do
        status_response="$(tnr status --no-wait --json)"
        status="$(json_instance_status "${INSTANCE_ID}" <<<"${status_response}" || true)"
        case "${status}" in
            RUNNING)
                log "Instance ${INSTANCE_ID} is RUNNING."
                return
                ;;
            FAILED|ERROR|DELETED|TERMINATED)
                die "Instance ${INSTANCE_ID} entered terminal state ${status}."
                ;;
            "")
                log "Instance ${INSTANCE_ID} is not visible yet; retrying."
                ;;
            *)
                log "Instance ${INSTANCE_ID} is ${status}; retrying."
                ;;
        esac
        sleep "${POLL_INTERVAL_SECONDS}"
    done
    die "Timed out waiting for instance ${INSTANCE_ID} to reach RUNNING."
}

configure_ssh() {
    log "Configuring SSH access for instance ${INSTANCE_ID}."
    tnr connect "${INSTANCE_ID}" --json >/dev/null
}

upload_bootstrap() {
    log "Uploading the private bootstrap with tnr scp."
    tnr scp "${DEPLOY_SCRIPT}" "${INSTANCE_ID}:${REMOTE_SCRIPT}"
}

start_remote_tmux() {
    local deploy_command remote_command ssh_alias
    local deploy_parts=(sudo bash "${REMOTE_SCRIPT}" "${DEPLOY_ARGS[@]}")

    printf -v deploy_command '%q ' "${deploy_parts[@]}"
    deploy_command="${deploy_command% } >${REMOTE_LOG} 2>&1"
    printf -v deploy_command '%q' "${deploy_command}"
    printf -v remote_command \
        'set -Eeuo pipefail; if ! command -v tmux >/dev/null 2>&1; then sudo env DEBIAN_FRONTEND=noninteractive apt-get update; sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y tmux; fi; chmod 700 %q; tmux has-session -t %q 2>/dev/null && { echo "tmux session already exists: %s" >&2; exit 1; }; tmux new-session -d -s %q bash -lc %s' \
        "${REMOTE_SCRIPT}" "${TMUX_SESSION}" "${TMUX_SESSION}" \
        "${TMUX_SESSION}" "${deploy_command}"

    ssh_alias="tnr-${INSTANCE_ID}"
    log "Starting deployment in tmux session ${TMUX_SESSION}."
    ssh "${ssh_alias}" "${remote_command}"
    log "Deployment started. Attach with: ssh -t ${ssh_alias} tmux attach -t ${TMUX_SESSION}"
    log "Bootstrap output: ssh ${ssh_alias} tail -f ${REMOTE_LOG}"
}

main() {
    parse_args "$@"
    validate
    check_authentication
    create_instance
    wait_until_running
    configure_ssh
    upload_bootstrap
    start_remote_tmux
}

main "$@"
