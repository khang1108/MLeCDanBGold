#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
umask 077

# Run the configured CPU embedding service and its existing named Cloudflare
# Tunnel as sibling processes. This script does not install dependencies or
# own tunnel routing; deploy_cloudflared_private.sh remains the private source
# for the tunnel token and Cloudflare dashboard configuration owns hostnames.

readonly SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly LLM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly REPO_DIR="$(cd -- "${LLM_DIR}/.." && pwd -P)"

HOST="127.0.0.1"
PORT="8100"
PYTHON_BIN="${REPO_DIR}/aic/bin/python"
CONFIG_FILE="${LLM_DIR}/config.yaml"
CREDENTIALS_FILE="${SCRIPT_DIR}/deploy_cloudflared_private.sh"
STARTUP_TIMEOUT_SECONDS="900"

UVICORN_PID=""
CLOUDFLARED_PID=""
TOKEN_FILE=""
CLEANUP_STARTED="false"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Run the HCMAI embedding API on CPU at localhost and connect the existing
Cloudflare named tunnel using the token in deploy_cloudflared_private.sh.

Options:
  --host HOST              Uvicorn bind host (default: 127.0.0.1)
  --port PORT              Uvicorn bind port (default: 8100)
  --python PATH            Python executable (default: aic/bin/python)
  --config PATH            Model YAML (default: llm/config.yaml)
  --credentials-file PATH  Private deployment script containing the tunnel token
  --startup-timeout SEC    Maximum local model startup wait (default: 900)
  -h, --help               Show this help

Enabled capabilities:
  visual embedding and caption/text embedding

Disabled capabilities:
  caption generation, OCR, ASR, diarization, reranking, VQA and query preparation

Examples:
  llm/scripts/${SCRIPT_NAME}
  llm/scripts/${SCRIPT_NAME} --port 8200

Stop both Uvicorn and cloudflared with Ctrl+C.
EOF
}

require_option_value() {
  local -r option="$1"
  local -r remaining="$2"
  [[ "${remaining}" -ge 2 ]] || die "${option} requires a value."
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --host)
        require_option_value "$1" "$#"
        HOST="$2"
        shift 2
        ;;
      --port)
        require_option_value "$1" "$#"
        PORT="$2"
        shift 2
        ;;
      --python)
        require_option_value "$1" "$#"
        PYTHON_BIN="$2"
        shift 2
        ;;
      --config)
        require_option_value "$1" "$#"
        CONFIG_FILE="$2"
        shift 2
        ;;
      --credentials-file)
        require_option_value "$1" "$#"
        CREDENTIALS_FILE="$2"
        shift 2
        ;;
      --startup-timeout)
        require_option_value "$1" "$#"
        STARTUP_TIMEOUT_SECONDS="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        [[ "$#" -eq 0 ]] || die "Positional arguments are not supported."
        ;;
      *)
        die "Unknown option: $1. Use --help."
        ;;
    esac
  done
}

require_command() {
  local -r command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "Required command not found: ${command_name}"
}

validate_inputs() {
  [[ "${PORT}" =~ ^[0-9]+$ ]] \
    && (( PORT >= 1 && PORT <= 65535 )) \
    || die "--port must be an integer from 1 to 65535."
  [[ "${STARTUP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
    || die "--startup-timeout must be a positive integer."
  [[ -x "${PYTHON_BIN}" ]] \
    || die "Python is not executable: ${PYTHON_BIN}"
  [[ -r "${CONFIG_FILE}" ]] \
    || die "Config is not readable: ${CONFIG_FILE}"
  [[ -r "${CREDENTIALS_FILE}" ]] \
    || die "Credentials file is not readable: ${CREDENTIALS_FILE}"

  require_command cloudflared
  require_command curl

  "${PYTHON_BIN}" -c 'import uvicorn' >/dev/null 2>&1 \
    || die "uvicorn is unavailable in ${PYTHON_BIN}."

  validate_cpu_config
}

validate_cpu_config() {
  "${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit(f"ERROR: Config must contain a YAML mapping: {path}")

for section in ("visual_embedding", "caption_embedding"):
    value = data.get(section)
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: Missing config section: {section}")
    if str(value.get("device", "")).strip().lower() != "cpu":
        raise SystemExit(f"ERROR: {section}.device must be 'cpu' in {path}")
PY
}

# Read one simple shell assignment without sourcing or executing the private
# deployment script. The tunnel token is deliberately never exported or logged.
read_private_assignment() {
  local -r key="$1"
  local line value=""

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" == "${key}="* ]] || continue
    value="${line#*=}"
    break
  done < "${CREDENTIALS_FILE}"

  if [[ "${#value}" -ge 2 && "${value}" == \"*\" && "${value}" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "${#value}" -ge 2 && "${value}" == \'*\' && "${value}" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi

  [[ -n "${value}" ]] || die "${key} is missing or empty in ${CREDENTIALS_FILE}."
  [[ "${value}" != *[[:space:]]* ]] \
    || die "${key} must be a single token without whitespace."
  printf '%s' "${value}"
}

create_token_file() {
  local tunnel_token
  tunnel_token="$(read_private_assignment CLOUDFLARE_TUNNEL_TOKEN)"
  TOKEN_FILE="$(mktemp "${TMPDIR:-/tmp}/hcmai-cloudflared-token.XXXXXX")"
  printf '%s\n' "${tunnel_token}" > "${TOKEN_FILE}"
  unset tunnel_token
}

terminate_process() {
  local -r pid="$1"
  local -r name="$2"

  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    log "Stopping ${name} (PID ${pid})"
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  [[ "${CLEANUP_STARTED}" == "false" ]] || return 0
  CLEANUP_STARTED="true"

  terminate_process "${CLOUDFLARED_PID}" cloudflared
  terminate_process "${UVICORN_PID}" Uvicorn

  if [[ -n "${TOKEN_FILE}" && -f "${TOKEN_FILE}" ]]; then
    rm -f -- "${TOKEN_FILE}"
  fi
}

handle_signal() {
  log "Shutdown requested"
  exit 130
}

on_error() {
  local -r exit_code="$1"
  local -r line_number="$2"
  log "Command failed at line ${line_number} (exit ${exit_code})"
}

trap 'on_error "$?" "$LINENO"' ERR
trap cleanup EXIT
trap handle_signal INT TERM

start_api() {
  log "Starting CPU embedding API at http://${HOST}:${PORT}"
  (
    cd "${REPO_DIR}"
    export CUDA_VISIBLE_DEVICES=""
    export HCMAI_LLM_CONFIG="${CONFIG_FILE}"
    export HCMAI_ENABLE_CAPTION="false"
    export HCMAI_ENABLE_OCR="false"
    export HCMAI_ENABLE_ASR="false"
    export HCMAI_ENABLE_DIARIZATION="false"
    export HCMAI_ENABLE_VISUAL_EMBEDDING="true"
    export HCMAI_ENABLE_CAPTION_EMBEDDING="true"
    export HCMAI_ENABLE_RERANKER="false"
    export HCMAI_ENABLE_VQA="false"
    export HCMAI_ENABLE_QUERY_PREPARATION="false"
    export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src"

    exec "${PYTHON_BIN}" -m uvicorn llm.server.api:app \
      --host "${HOST}" \
      --port "${PORT}" \
      --workers 1 \
      --proxy-headers \
      --forwarded-allow-ips=127.0.0.1
  ) &
  UVICORN_PID="$!"
}

wait_for_api() {
  local attempt
  for ((attempt = 1; attempt <= STARTUP_TIMEOUT_SECONDS; attempt++)); do
    if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
      wait "${UVICORN_PID}" || true
      die "Uvicorn exited before the API became ready."
    fi
    if curl --fail --silent --show-error \
      --max-time 2 "http://${HOST}:${PORT}/ready" >/dev/null 2>&1; then
      log "CPU models are ready"
      return 0
    fi
    sleep 1
  done

  die "API was not ready after ${STARTUP_TIMEOUT_SECONDS} seconds."
}

start_tunnel() {
  log "Starting the existing Cloudflare Tunnel"
  cloudflared tunnel --no-autoupdate run --token-file "${TOKEN_FILE}" &
  CLOUDFLARED_PID="$!"

  sleep 1
  kill -0 "${CLOUDFLARED_PID}" 2>/dev/null \
    || { wait "${CLOUDFLARED_PID}" || true; die "cloudflared exited during startup."; }
}

wait_for_process_exit() {
  local exit_code=0

  log "Local API: http://${HOST}:${PORT}/docs"
  log "Tunnel route: use the hostname configured for port ${PORT} in Cloudflare"
  log "Press Ctrl+C to stop both processes"

  set +e
  wait -n "${UVICORN_PID}" "${CLOUDFLARED_PID}"
  exit_code="$?"
  set -e

  if [[ "${exit_code}" -eq 0 ]]; then
    log "A child process stopped normally; shutting down its sibling"
  else
    log "A child process failed with exit ${exit_code}; shutting down its sibling"
  fi
  return "${exit_code}"
}

main() {
  parse_args "$@"
  validate_inputs
  create_token_file
  start_api
  wait_for_api
  start_tunnel
  wait_for_process_exit
}

main "$@"
