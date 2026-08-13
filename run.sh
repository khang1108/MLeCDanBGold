#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly THUNDER_LAUNCHER="${REPO_ROOT}/llm/launch_thunder_instance.sh"
readonly REMOTE_LLM_LOG="/opt/hcmai/logs/llm.log"

LOCAL_HOST="${HCMAI_LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${HCMAI_LOCAL_PORT:-8000}"
LOCAL_LOG_LEVEL="${HCMAI_LOG_LEVEL:-INFO}"
LOCAL_LOG_LEVEL="${LOCAL_LOG_LEVEL^^}"
UVICORN_LOG_LEVEL="${HCMAI_UVICORN_LOG_LEVEL:-info}"
UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL,,}"
PYTHON_BIN="${HCMAI_PYTHON_BIN:-${REPO_ROOT}/aic/bin/python}"
LOG_RETRY_SECONDS="${HCMAI_LOG_RETRY_SECONDS:-5}"

INSTANCE_ID_FILE=""
CONNECTION_INFO_FILE=""
REMOTE_LOG_PID=""

log() {
    printf '[run %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  ./run.sh (--gpu a6000|l40 | --instance ID) [launcher options] -- [model options]

This command:
  1. creates or reuses and bootstraps a Thunder Compute instance;
  2. follows /opt/hcmai/logs/llm.log as [thunder:llm];
  3. runs the local API from src/hcmai/app.py with Uvicorn.

Local configuration (environment variables):
  HCMAI_LOCAL_HOST          Uvicorn host (default: 127.0.0.1).
  HCMAI_LOCAL_PORT          Uvicorn port (default: 8000).
  HCMAI_LOG_LEVEL           src/hcmai logger level (default: INFO).
  HCMAI_UVICORN_LOG_LEVEL   Uvicorn log level (default: info).
  HCMAI_UVICORN_RELOAD      Set to true to enable --reload.
  HCMAI_LOG_RETRY_SECONDS   Remote-log reconnect delay (default: 5).

All command-line arguments are forwarded to launch_thunder_instance.sh.

Example:
  ./run.sh --gpu l40 --token "$TNR_TOKEN" -- \
      --caption true --caption-embedding true

Recover an existing instance after a local launcher failure:
  ./run.sh --instance 0 -- --caption-embedding true --reranker true --vqa true

Press Ctrl-C to stop local Uvicorn and log streaming. The Thunder instance and
its remote services are intentionally left running.
EOF
}

help_requested() {
    local argument
    for argument in "$@"; do
        [[ "${argument}" == "--" ]] && return 1
        [[ "${argument}" == "--help" || "${argument}" == "-h" ]] && return 0
    done
    return 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate() {
    [[ -x "${THUNDER_LAUNCHER}" ]] \
        || die "Thunder launcher is missing or not executable: ${THUNDER_LAUNCHER}"
    [[ -x "${PYTHON_BIN}" ]] || die "Python is missing or not executable: ${PYTHON_BIN}"
    command -v ssh >/dev/null 2>&1 || die "Required command not found: ssh"
    [[ -n "${LOCAL_HOST}" ]] || die "HCMAI_LOCAL_HOST must not be empty."
    is_positive_integer "${LOCAL_PORT}" \
        || die "HCMAI_LOCAL_PORT must be a positive integer."
    (( LOCAL_PORT <= 65535 )) || die "HCMAI_LOCAL_PORT must be at most 65535."
    is_positive_integer "${LOG_RETRY_SECONDS}" \
        || die "HCMAI_LOG_RETRY_SECONDS must be a positive integer."
    [[ "${LOCAL_LOG_LEVEL}" =~ ^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$ ]] \
        || die "HCMAI_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
    [[ "${UVICORN_LOG_LEVEL}" =~ ^(critical|error|warning|info|debug|trace)$ ]] \
        || die "HCMAI_UVICORN_LOG_LEVEL is invalid."
    [[ "${HCMAI_UVICORN_RELOAD:-false}" == "true" \
        || "${HCMAI_UVICORN_RELOAD:-false}" == "false" ]] \
        || die "HCMAI_UVICORN_RELOAD must be true or false."
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${REMOTE_LOG_PID}" ]] \
        && kill -0 "${REMOTE_LOG_PID}" 2>/dev/null; then
        kill -TERM "${REMOTE_LOG_PID}" 2>/dev/null || true
        wait "${REMOTE_LOG_PID}" 2>/dev/null || true
    fi
    if [[ -n "${INSTANCE_ID_FILE}" && -f "${INSTANCE_ID_FILE}" ]]; then
        rm -f -- "${INSTANCE_ID_FILE}"
    fi
    if [[ -n "${CONNECTION_INFO_FILE}" && -f "${CONNECTION_INFO_FILE}" ]]; then
        rm -f -- "${CONNECTION_INFO_FILE}"
    fi
    exit "${exit_code}"
}

follow_remote_llm_log() {
    local instance_id=$1
    local ssh_host=$2
    local ssh_port=$3
    local ssh_key_file=$4
    local ssh_pid=""

    trap '
        if [[ -n "${ssh_pid}" ]]; then
            kill -TERM "${ssh_pid}" 2>/dev/null || true
            wait "${ssh_pid}" 2>/dev/null || true
        fi
        exit 0
    ' TERM INT

    while true; do
        log "Following Thunder instance ${instance_id} log ${REMOTE_LLM_LOG}."
        ssh \
            -T \
            -i "${ssh_key_file}" \
            -p "${ssh_port}" \
            -o BatchMode=yes \
            -o IdentitiesOnly=yes \
            -o PasswordAuthentication=no \
            -o KbdInteractiveAuthentication=no \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -o LogLevel=ERROR \
            -o ConnectTimeout=30 \
            "ubuntu@${ssh_host}" \
            "sudo tail -n 100 -F '${REMOTE_LLM_LOG}' | sed -u 's/^/[thunder:llm] /'" &
        ssh_pid=$!
        if wait "${ssh_pid}"; then
            log "Remote llm.log stream ended; reconnecting in ${LOG_RETRY_SECONDS}s."
        else
            log "Remote llm.log unavailable; reconnecting in ${LOG_RETRY_SECONDS}s."
        fi
        ssh_pid=""
        sleep "${LOG_RETRY_SECONDS}"
    done
}

run_local_api() {
    local uvicorn_args=(
        -m uvicorn hcmai.app:app
        --host "${LOCAL_HOST}"
        --port "${LOCAL_PORT}"
        --log-level "${UVICORN_LOG_LEVEL}"
    )
    if [[ "${HCMAI_UVICORN_RELOAD:-false}" == "true" ]]; then
        uvicorn_args+=(--reload)
    fi

    log "Starting local API at http://${LOCAL_HOST}:${LOCAL_PORT}."
    log "Local hcmai logger level: ${LOCAL_LOG_LEVEL}."
    cd "${REPO_ROOT}"
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    HCMAI_LOG_LEVEL="${LOCAL_LOG_LEVEL}" \
        "${PYTHON_BIN}" "${uvicorn_args[@]}"
}

main() {
    if (( $# == 0 )); then
        usage
        exit 1
    fi
    if help_requested "$@"; then
        usage
        printf '\nThunder launcher options:\n\n'
        "${THUNDER_LAUNCHER}" --help
        exit 0
    fi

    validate
    INSTANCE_ID_FILE="$(mktemp /tmp/hcmai-thunder-instance-id.XXXXXX)"
    CONNECTION_INFO_FILE="$(mktemp /tmp/hcmai-thunder-connection.XXXXXX)"
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    log "Launching the Thunder Compute model service."
    HCMAI_THUNDER_INSTANCE_ID_FILE="${INSTANCE_ID_FILE}" \
    HCMAI_THUNDER_CONNECTION_FILE="${CONNECTION_INFO_FILE}" \
        "${THUNDER_LAUNCHER}" "$@"

    local instance_id
    instance_id="$(<"${INSTANCE_ID_FILE}")"
    [[ "${instance_id}" =~ ^[0-9]+$ ]] \
        || die "Launcher did not return a valid Thunder instance ID."
    log "Thunder instance ID: ${instance_id}."

    local -a connection_fields=()
    mapfile -t connection_fields < "${CONNECTION_INFO_FILE}"
    (( ${#connection_fields[@]} == 3 )) \
        || die "Launcher did not return complete SSH connection information."
    local ssh_host="${connection_fields[0]}"
    local ssh_port="${connection_fields[1]}"
    local ssh_key_file="${connection_fields[2]}"

    follow_remote_llm_log \
        "${instance_id}" "${ssh_host}" "${ssh_port}" "${ssh_key_file}" &
    REMOTE_LOG_PID=$!
    run_local_api
}

main "$@"
