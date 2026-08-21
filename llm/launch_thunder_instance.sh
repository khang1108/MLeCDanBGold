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
SSH_HOST=""
SSH_PORT=""
SSH_KEY_FILE=""

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    if [[ -n "${INSTANCE_ID}" ]]; then
        printf 'ERROR: Thunder instance %s may still be billable.\n' \
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
        printf 'ERROR: Thunder instance %s may still be billable.\n' \
            "${INSTANCE_ID}" >&2
    fi
    exit "${exit_code}"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Usage:
  llm/launch_thunder_instance.sh (--gpu a6000|l40 | --instance ID) [options] \
      -- [deploy_cloudflared_private.sh options]

Instance selection (choose one):
  --gpu GPU               GPU type: a6000 or l40.
  --instance ID           Reuse an existing Thunder instance by positional ID.

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

Bootstrap service options (place after --; reranker and VQA default to false):
  --caption true|false            Florence-2 frame captioning.
  --ocr true|false                OCR model service.
  --asr true|false                Qwen3-ASR service.
  --visual-embedding true|false   SigLIP2 image embedding service.
  --caption-embedding true|false  BGE-M3 text embedding service.
  --reranker true|false           Qwen3-VL visual reranking (default: false).
  --vqa true|false                Qwen2.5-VL grounded VQA (default: false).

At least one bootstrap service must remain enabled. Everything after -- is
forwarded unchanged to deploy_cloudflared_private.sh on the instance.

Examples:
  # Start caption, OCR, and ASR using their defaults.
  llm/launch_thunder_instance.sh --gpu l40 --token "$TOKEN" --

  # Start only caption and OCR.
  llm/launch_thunder_instance.sh --gpu l40 --token "$TOKEN" -- --asr false

  # Start SigLIP/BGE embeddings, Qwen reranking, and grounded VQA.
  llm/launch_thunder_instance.sh --gpu l40 --token "$TOKEN" -- \
    --caption false --ocr false --asr false --visual-embedding true \
    --caption-embedding true --reranker true --vqa true
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
            --instance)
                require_value "$1" "$#"
                INSTANCE_ID=$2
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
    if [[ -n "${INSTANCE_ID}" ]]; then
        [[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] \
            || die "--instance must be a non-negative integer."
        [[ -z "${GPU}" ]] || die "Use either --gpu or --instance, not both."
    else
        [[ "${GPU}" == "a6000" || "${GPU}" == "l40" ]] \
            || die "--gpu must be either a6000 or l40 when --instance is omitted."
    fi
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
    for command_name in tnr python3 ssh; do
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
        || die "Thunder CLI is not logged in; provide --token TOKEN or set TNR_API_TOKEN."

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

json_instance_connection() {
    local expected_id=$1
    python3 -c '
import json
import sys

expected = sys.argv[1]
payload = json.load(sys.stdin)

def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)

def first_value(value, keys):
    for candidate in walk(value):
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            result = candidate.get(key)
            if result not in (None, ""):
                return result
    return None

for candidate in walk(payload):
    if not isinstance(candidate, dict):
        continue
    identifier = candidate.get("identifier", candidate.get("instance_id", candidate.get("id")))
    if identifier is None or str(identifier) != expected:
        continue
    uuid = first_value(candidate, ("uuid", "instance_uuid"))
    host = first_value(candidate, ("ip", "ip_address", "public_ip", "publicIp"))
    port = first_value(candidate, ("port", "ssh_port", "sshPort"))
    if uuid is None or host is None:
        continue
    print(uuid)
    print(host)
    print(22 if port is None else port)
    raise SystemExit(0)

raise SystemExit("status response did not contain SSH connection metadata for the instance")
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
    log "Created Thunder instance ${INSTANCE_ID}."
}

publish_instance_id() {
    if [[ -n "${HCMAI_THUNDER_INSTANCE_ID_FILE:-}" ]]; then
        [[ -f "${HCMAI_THUNDER_INSTANCE_ID_FILE}" \
            && ! -L "${HCMAI_THUNDER_INSTANCE_ID_FILE}" ]] \
            || die "Instance-ID destination must be an existing regular file."
        printf '%s\n' "${INSTANCE_ID}" \
            > "${HCMAI_THUNDER_INSTANCE_ID_FILE}"
    fi
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

upload_bootstrap() {
    log "Uploading the private bootstrap with tnr scp."
    tnr scp "${DEPLOY_SCRIPT}" "${INSTANCE_ID}:${REMOTE_SCRIPT}"
}

resolve_ssh_connection() {
    local status_response connection_info instance_uuid
    local -a connection_fields=() key_candidates=()

    status_response="$(tnr status --no-wait --json)"
    connection_info="$(json_instance_connection "${INSTANCE_ID}" <<<"${status_response}")"
    mapfile -t connection_fields <<<"${connection_info}"
    (( ${#connection_fields[@]} == 3 )) \
        || die "Thunder returned incomplete SSH connection metadata."

    instance_uuid="${connection_fields[0]}"
    SSH_HOST="${connection_fields[1]}"
    SSH_PORT="${connection_fields[2]}"
    [[ "${instance_uuid}" =~ ^[A-Za-z0-9_-]+$ ]] \
        || die "Thunder returned an invalid instance UUID."
    [[ "${SSH_HOST}" =~ ^[A-Za-z0-9._:-]+$ ]] \
        || die "Thunder returned an invalid SSH host."
    is_positive_integer "${SSH_PORT}" \
        || die "Thunder returned an invalid SSH port."
    (( SSH_PORT <= 65535 )) || die "Thunder returned an invalid SSH port."

    if [[ -n "${TNR_HOME:-}" ]]; then
        key_candidates+=("${TNR_HOME}/keys/${instance_uuid}")
    else
        [[ -n "${HOME:-}" ]] \
            && key_candidates+=("${HOME}/.thunder/keys/${instance_uuid}")
        if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
            key_candidates+=("${XDG_CACHE_HOME}/thunder/keys/${instance_uuid}")
        elif [[ -n "${HOME:-}" ]]; then
            key_candidates+=("${HOME}/.cache/thunder/keys/${instance_uuid}")
        fi
        key_candidates+=("/tmp/thunder-$(id -u)/keys/${instance_uuid}")
    fi

    local candidate
    for candidate in "${key_candidates[@]}"; do
        if [[ -f "${candidate}" && -r "${candidate}" ]]; then
            SSH_KEY_FILE="${candidate}"
            break
        fi
    done
    [[ -n "${SSH_KEY_FILE}" ]] \
        || die "Thunder SSH key was not cached by tnr scp for instance ${INSTANCE_ID}."

    if [[ -n "${HCMAI_THUNDER_CONNECTION_FILE:-}" ]]; then
        [[ -f "${HCMAI_THUNDER_CONNECTION_FILE}" \
            && ! -L "${HCMAI_THUNDER_CONNECTION_FILE}" ]] \
            || die "Connection-info destination must be an existing regular file."
        printf '%s\n%s\n%s\n' "${SSH_HOST}" "${SSH_PORT}" "${SSH_KEY_FILE}" \
            > "${HCMAI_THUNDER_CONNECTION_FILE}"
    fi
}

ssh_command() {
    ssh \
        -T \
        -i "${SSH_KEY_FILE}" \
        -p "${SSH_PORT}" \
        -o BatchMode=yes \
        -o IdentitiesOnly=yes \
        -o PasswordAuthentication=no \
        -o KbdInteractiveAuthentication=no \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        -o ConnectTimeout=30 \
        "ubuntu@${SSH_HOST}" "$@"
}

start_remote_tmux() {
    local deploy_command remote_command
    local deploy_parts=(sudo bash "${REMOTE_SCRIPT}" "${DEPLOY_ARGS[@]}")

    printf -v deploy_command '%q ' "${deploy_parts[@]}"
    deploy_command="${deploy_command% } >${REMOTE_LOG} 2>&1"
    printf -v deploy_command '%q' "${deploy_command}"
    printf -v remote_command \
        'set -Eeuo pipefail; if ! command -v tmux >/dev/null 2>&1; then sudo env DEBIAN_FRONTEND=noninteractive apt-get update; sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y tmux; fi; chmod 700 %q; tmux has-session -t %q 2>/dev/null && { echo "tmux session already exists: %s" >&2; exit 1; }; tmux new-session -d -s %q bash -lc %s' \
        "${REMOTE_SCRIPT}" "${TMUX_SESSION}" "${TMUX_SESSION}" \
        "${TMUX_SESSION}" "${deploy_command}"

    log "Starting deployment in tmux session ${TMUX_SESSION}."
    ssh_command "${remote_command}"
    log "Deployment started for instance ${INSTANCE_ID}."
    log "Use run.sh log streaming and tnr status to inspect the deployment."
}

main() {
    parse_args "$@"
    validate
    check_authentication
    if [[ -z "${INSTANCE_ID}" ]]; then
        create_instance
    else
        log "Reusing Thunder instance ${INSTANCE_ID}."
    fi
    publish_instance_id
    wait_until_running
    upload_bootstrap
    resolve_ssh_connection
    start_remote_tmux
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
