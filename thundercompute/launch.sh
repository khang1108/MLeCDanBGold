#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Provision one disposable GPU VM and keep this process attached to its
# lifecycle.  The private bootstrap is deliberately supplied at runtime; it
# is never copied into this image or committed to the repository.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly DEFAULT_STATE_FILE="${REPO_ROOT}/.thundercompute/instance-id"
readonly DEFAULT_DEPLOY_SCRIPT="${SCRIPT_DIR}/deploy_cloudflared_private.sh"
readonly REMOTE_SCRIPT="${HCMAI_THUNDER_REMOTE_SCRIPT:-/home/ubuntu/deploy_cloudflared_private.sh}"
readonly REMOTE_LOG="${HCMAI_THUNDER_REMOTE_LOG:-/tmp/hcmai-deploy.log}"

GPU="${HCMAI_THUNDER_GPU:-l40}"
GPU_FROM_ARG=false
TEMPLATE="${HCMAI_THUNDER_TEMPLATE:-base}"
VCPUS="${HCMAI_THUNDER_VCPUS:-8}"
PRIMARY_DISK_GB="${HCMAI_THUNDER_DISK_GB:-${HCMAI_THUNDER_PRIMARY_DISK_GB:-200}}"
TOKEN=""
WAIT_TIMEOUT_SECONDS="${HCMAI_THUNDER_WAIT_TIMEOUT_SECONDS:-900}"
POLL_INTERVAL_SECONDS="${HCMAI_THUNDER_POLL_INTERVAL_SECONDS:-10}"
TMUX_SESSION="${HCMAI_THUNDER_TMUX_SESSION:-hcmai-deploy}"
DEPLOY_SCRIPT="${HCMAI_THUNDER_DEPLOY_SCRIPT:-${DEFAULT_DEPLOY_SCRIPT}}"
INSTANCE_ID="${HCMAI_THUNDER_INSTANCE_ID:-}"
INSTANCE_ID_FILE="${HCMAI_THUNDER_INSTANCE_ID_FILE:-${DEFAULT_STATE_FILE}}"
DELETE_ON_EXIT="${HCMAI_THUNDER_DELETE_ON_EXIT:-true}"
DELETE_REUSED="${HCMAI_THUNDER_DELETE_REUSED:-false}"
DEPLOY_ARGS=()
INSTANCE_CREATED=false
SSH_HOST=""
SSH_PORT=""
SSH_KEY_FILE=""

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    trap - ERR
    printf 'ERROR: command failed at line %s (exit %s).\n' \
        "${BASH_LINENO[0]}" "${exit_code}" >&2
    exit "${exit_code}"
}
trap on_error ERR

cleanup() {
    local exit_code=$?
    local should_delete=false
    trap - ERR EXIT INT TERM

    if [[ "${DELETE_ON_EXIT}" == "true" && -n "${INSTANCE_ID}" ]]; then
        if [[ "${INSTANCE_CREATED}" == "true" || "${DELETE_REUSED}" == "true" ]]; then
            should_delete=true
        fi
    fi

    if [[ "${should_delete}" == "true" ]]; then
        log "Deleting Thunder instance ${INSTANCE_ID}."
        if tnr delete --yes "${INSTANCE_ID}" </dev/null; then
            clear_instance_state
            log "Thunder instance ${INSTANCE_ID} deleted."
        else
            printf 'ERROR: unable to delete Thunder instance %s; it may remain billable.\n' \
                "${INSTANCE_ID}" >&2
            (( exit_code == 0 )) && exit_code=1
        fi
    fi

    return "${exit_code}"
}
trap cleanup EXIT

on_signal() {
    local signal=$1
    log "Received ${signal}; stopping the local controller."
    exit 143
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

usage() {
    cat <<'EOF'
Usage:
  thundercompute/launch.sh (--gpu GPU | --instance ID) [options] \
      -- [deploy_cloudflared_private.sh options]

Instance selection:
  --gpu GPU                 Create a new instance, for example l40 or a6000.
  --instance ID             Reuse an existing Thunder instance.

Instance options:
  --template NAME           Thunder template (default: base).
  --vcpus COUNT             vCPU count (default: 8).
  --disk GB                 Primary disk size (passed to the installed tnr CLI).
  --wait-timeout SECONDS    Maximum time waiting for RUNNING (default: 900).
  --poll-interval SECONDS   Status polling interval (default: 10).
  --tmux-session NAME       Remote deployment session (default: hcmai-deploy).
  --token TOKEN             TNR API token; TNR_API_TOKEN is also supported.
  --keep                    Do not delete the instance when this process exits.
  --delete-reused           Also delete an instance passed with --instance.
  --help                    Show this help.

The private bootstrap is uploaded with:
  tnr scp SCRIPT ID:/home/ubuntu/

There is no documented non-interactive `tnr exec` command.  After SCP, this
launcher uses the SSH metadata returned by `tnr status --json` and starts the
bootstrap in a detached remote tmux session.

The controller stays attached after deployment.  Use SIGTERM, Ctrl-C, or
`docker compose stop thundercompute` for cleanup.  SIGKILL cannot be trapped;
use thundercompute/delete.sh with the saved instance ID after a force
kill.
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
                GPU_FROM_ARG=true
                shift 2
                ;;
            --instance)
                require_value "$1" "$#"
                INSTANCE_ID=$2
                shift 2
                ;;
            --template)
                require_value "$1" "$#"
                TEMPLATE=$2
                shift 2
                ;;
            --vcpus)
                require_value "$1" "$#"
                VCPUS=$2
                shift 2
                ;;
            --primary-disk|--disk)
                require_value "$1" "$#"
                PRIMARY_DISK_GB=$2
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
            --token)
                require_value "$1" "$#"
                TOKEN=$2
                shift 2
                ;;
            --keep)
                DELETE_ON_EXIT=false
                shift
                ;;
            --delete-reused)
                DELETE_REUSED=true
                shift
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

validate() {
    if [[ -n "${INSTANCE_ID}" ]]; then
        [[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] \
            || die "--instance must be a non-negative integer."
        [[ "${GPU_FROM_ARG}" == "false" ]] \
            || die "Use either --gpu or --instance, not both."
    else
        [[ -n "${GPU}" ]] || die "--gpu is required when --instance is omitted."
    fi
    is_positive_integer "${VCPUS}" || die "--vcpus must be a positive integer."
    is_positive_integer "${PRIMARY_DISK_GB}" \
        || die "--disk must be a positive integer."
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

    mkdir -p "$(dirname -- "${INSTANCE_ID_FILE}")"
    [[ ! -L "${INSTANCE_ID_FILE}" ]] \
        || die "Instance-ID state path must not be a symbolic link."
}

check_authentication() {
    load_token_from_file

    if tnr status --no-wait --json >/dev/null 2>&1; then
        log "Thunder CLI is already authenticated."
        return
    fi

    if [[ -z "${TOKEN}" ]]; then
        TOKEN="${TNR_API_TOKEN:-}"
    fi
    [[ -n "${TOKEN}" ]] \
        || die "Thunder CLI is not logged in; provide --token TOKEN or set TNR_API_TOKEN/TNR_API_TOKEN_FILE."

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

write_instance_state() {
    printf '%s\n' "${INSTANCE_ID}" > "${INSTANCE_ID_FILE}"
}

clear_instance_state() {
    [[ -e "${INSTANCE_ID_FILE}" ]] || return 0
    [[ ! -L "${INSTANCE_ID_FILE}" ]] \
        || die "Instance-ID state path became a symbolic link."
    : > "${INSTANCE_ID_FILE}"
}

create_instance() {
    local create_response
    log "Creating Thunder instance: gpu=${GPU}, vcpus=${VCPUS}, disk=${PRIMARY_DISK_GB}GB."
    create_response="$(tnr create \
        --gpu "${GPU}" \
        --num-gpus 1 \
        --vcpus "${VCPUS}" \
        --template "${TEMPLATE}" \
        --disk "${PRIMARY_DISK_GB}" \
        --json \
        --yes)"
    INSTANCE_ID="$(json_instance_id <<<"${create_response}")"
    [[ "${INSTANCE_ID}" =~ ^[0-9]+$ ]] \
        || die "Thunder returned an invalid instance ID: ${INSTANCE_ID}"
    INSTANCE_CREATED=true
    write_instance_state
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

upload_bootstrap() {
    log "Uploading the private bootstrap with tnr scp to /home/ubuntu/."
    tnr scp "${DEPLOY_SCRIPT}" "${INSTANCE_ID}:/home/ubuntu/" --yes
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

    if [[ -n "${HCMAI_THUNDER_SSH_KEY_FILE:-}" ]]; then
        key_candidates+=("${HCMAI_THUNDER_SSH_KEY_FILE}")
    elif [[ -n "${TNR_HOME:-}" ]]; then
        key_candidates+=("${TNR_HOME}/keys/${instance_uuid}")
    else
        [[ -n "${HOME:-}" ]] && key_candidates+=("${HOME}/.thunder/keys/${instance_uuid}")
        [[ -n "${HOME:-}" ]] && key_candidates+=("${HOME}/.tnr/keys/${instance_uuid}")
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
        || die "Thunder SSH key was not cached by tnr scp for instance ${INSTANCE_ID}. Set HCMAI_THUNDER_SSH_KEY_FILE if needed."

    if [[ -n "${HCMAI_THUNDER_CONNECTION_FILE:-}" ]]; then
        [[ ! -L "${HCMAI_THUNDER_CONNECTION_FILE}" ]] \
            || die "Connection-info destination must not be a symbolic link."
        mkdir -p "$(dirname -- "${HCMAI_THUNDER_CONNECTION_FILE}")"
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
        'set -Eeuo pipefail; chmod 700 %q; if ! command -v tmux >/dev/null 2>&1; then sudo env DEBIAN_FRONTEND=noninteractive apt-get update; sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y tmux; fi; tmux has-session -t %q 2>/dev/null && { echo "tmux session already exists: %s" >&2; exit 1; }; tmux new-session -d -s %q bash -lc %s' \
        "${REMOTE_SCRIPT}" "${TMUX_SESSION}" "${TMUX_SESSION}" \
        "${TMUX_SESSION}" "${deploy_command}"

    log "Starting deployment in remote tmux session ${TMUX_SESSION}."
    ssh_command "${remote_command}"
    log "Deployment started for instance ${INSTANCE_ID}."
    log "Remote log: ${REMOTE_LOG}; local stop will delete the instance."
}

wait_for_controller_stop() {
    log "Controller is attached to instance ${INSTANCE_ID}."
    log "Send SIGTERM/Ctrl-C to delete it; use delete.sh after SIGKILL."
    while :; do
        sleep 3600
    done
}

main() {
    parse_args "$@"
    validate
    check_authentication
    if [[ -z "${INSTANCE_ID}" ]]; then
        create_instance
    else
        log "Reusing Thunder instance ${INSTANCE_ID}."
        write_instance_state
    fi
    wait_until_running
    upload_bootstrap
    resolve_ssh_connection
    start_remote_tmux
    wait_for_controller_stop
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
