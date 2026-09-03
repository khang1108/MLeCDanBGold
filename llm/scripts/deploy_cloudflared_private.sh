#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# PRIVATE LOCAL BOOTSTRAP: this ignored file is copied directly to each
# throwaway Thunder VM. Paste the tunnel token once here; never commit it.
REPO_URL="https://github.com/khang1108/MLeCDanBGold.git"
REPO_BRANCH="main"
INSTALL_ROOT="/opt/hcmai"
REPO_DIR="${INSTALL_ROOT}/repo"
SUPERVISOR_ROOT="${INSTALL_ROOT}/supervisor"
SUPERVISORD_BIN="${SUPERVISOR_ROOT}/bin/supervisord"
SUPERVISORCTL_BIN="${SUPERVISOR_ROOT}/bin/supervisorctl"
SUPERVISOR_VERSION="4.3.0"
SERVICE_USER="hcmai"
# Qwen reranking/VQA and Qwen3-ASR currently require incompatible Transformers
# ranges. Keep the API and ASR environments separate when both are enabled.
API_VENV="${REPO_DIR}/aic"
ASR_VENV="${REPO_DIR}/aic-asr"
API_PYTHON="${API_VENV}/bin/python"
ASR_PYTHON="${API_PYTHON}"

API_HOST="127.0.0.1"
API_PORT="8100"
API_PUBLIC_HOSTNAME="api.iamphuckhang.dev"
ASR_HOST="127.0.0.1"
ASR_PORT="8101"
ASR_PUBLIC_HOSTNAME="asr.iamphuckhang.dev"

CLOUDFLARE_TUNNEL_TOKEN="eyJhIjoiZDY1ZDUwM2E1NWM3YWFhODZjYjI3OWU1NzEzMTkyOWMiLCJ0IjoiYjAyOGU0MmMtZjRjNi00NmQ1LWIzYjAtZjgzNzEzMjg3NTZlIiwicyI6IllUZzVNR1ZqT0dZdFpURTBZeTAwTTJJMExXRTRNR0l0TmpFd1pEQXlORFk0TURObCJ9"
HCMAI_CF_ACCESS_CLIENT_ID="e4c8fef0e7edaa4a0eddafe78a39fb37.access"
HCMAI_CF_ACCESS_CLIENT_SECRET="0b71d885488b3ddf69f27d68d23258956dabdb7fce9916405920b81c7608266d"
ENABLE_CAPTION="true"
ENABLE_OCR="true"
ENABLE_ASR="true"
ENABLE_VISUAL_EMBEDDING="true"
ENABLE_CAPTION_EMBEDDING="true"
ENABLE_RERANKER="false"
ENABLE_VQA="false"
ENABLE_QUERY_PREPARATION="false"

API_SERVICE="hcmai-caption-ocr"
ASR_SERVICE="hcmai-asr"
TUNNEL_SERVICE="hcmai-cloudflared"
MODEL_SERVICES=()

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: sudo bash deploy_cloudflared_private.sh [service options]

Service options:
  --caption true|false
  --ocr true|false
  --asr true|false
  --visual-embedding true|false
  --caption-embedding true|false
  --reranker true|false       Qwen3-VL visual reranking (default: false)
  --vqa true|false            Qwen2.5-VL grounded VQA (default: false)
  --query-preparation true|false  Qwen3-4B query preparation (default: false)
  --help

Public routes configured on the same Cloudflare tunnel:
  api.iamphuckhang.dev -> http://127.0.0.1:8100 (caption + OCR + embeddings + reranker + VQA + query preparation)
  asr.iamphuckhang.dev -> http://127.0.0.1:8101 (ASR)
EOF
}

boolean() {
    [[ "$2" == "true" || "$2" == "false" ]] \
        || die "$1 must be true or false."
    printf '%s' "$2"
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --caption|--ocr|--asr|--visual-embedding|--caption-embedding|--reranker|--vqa|--query-preparation)
                [[ "$#" -ge 2 ]] || die "$1 requires true or false."
                local value
                value="$(boolean "$1" "$2")"
                case "$1" in
                    --caption) ENABLE_CAPTION="${value}" ;;
                    --ocr) ENABLE_OCR="${value}" ;;
                    --asr) ENABLE_ASR="${value}" ;;
                    --visual-embedding) ENABLE_VISUAL_EMBEDDING="${value}" ;;
                    --caption-embedding) ENABLE_CAPTION_EMBEDDING="${value}" ;;
                    --reranker) ENABLE_RERANKER="${value}" ;;
                    --vqa) ENABLE_VQA="${value}" ;;
                    --query-preparation) ENABLE_QUERY_PREPARATION="${value}" ;;
                esac
                shift 2
                ;;
            --help) usage; exit 0 ;;
            *) die "Unknown option: $1. Use --help." ;;
        esac
    done
}

validate() {
    [[ "${EUID}" -eq 0 ]] || die "Run with: sudo bash $0"
    [[ -n "${CLOUDFLARE_TUNNEL_TOKEN}" \
        && "${CLOUDFLARE_TUNNEL_TOKEN}" != REPLACE_WITH_* ]] \
        || die "Paste the eyJ... token into CLOUDFLARE_TUNNEL_TOKEN."
    [[ "${CLOUDFLARE_TUNNEL_TOKEN}" != *$'\n'* ]] \
        || die "CLOUDFLARE_TUNNEL_TOKEN must be one line."
    [[ "${API_PUBLIC_HOSTNAME}" == *.* ]] || die "Invalid API_PUBLIC_HOSTNAME."
    [[ "${ASR_PUBLIC_HOSTNAME}" == *.* ]] || die "Invalid ASR_PUBLIC_HOSTNAME."
    if [[ -z "${HCMAI_CF_ACCESS_CLIENT_ID}" \
        && -n "${HCMAI_CF_ACCESS_CLIENT_SECRET}" ]] \
        || [[ -n "${HCMAI_CF_ACCESS_CLIENT_ID}" \
        && -z "${HCMAI_CF_ACCESS_CLIENT_SECRET}" ]]; then
        die "Set both Cloudflare Access credentials or neither."
    fi
    [[ "${ENABLE_CAPTION}" == "true" \
        || "${ENABLE_OCR}" == "true" \
        || "${ENABLE_ASR}" == "true" \
        || "${ENABLE_VISUAL_EMBEDDING}" == "true" \
        || "${ENABLE_CAPTION_EMBEDDING}" == "true" \
        || "${ENABLE_RERANKER}" == "true" \
        || "${ENABLE_VQA}" == "true" \
        || "${ENABLE_QUERY_PREPARATION}" == "true" ]] \
        || die "Enable at least one model. Use --help."
}

install_system_dependencies() {
    log "Installing system dependencies"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y ca-certificates curl git jq python3 python3-venv supervisor
    python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("HCMAI requires Python >=3.11.")
PY
}

install_supervisor() {
    log "Installing Python 3.12-compatible Supervisor ${SUPERVISOR_VERSION}"
    [[ -x "${SUPERVISOR_ROOT}/bin/python" ]] \
        || python3 -m venv "${SUPERVISOR_ROOT}"
    "${SUPERVISOR_ROOT}/bin/python" -m pip install --upgrade \
        "supervisor==${SUPERVISOR_VERSION}"
    ln -sfn "${SUPERVISORD_BIN}" /usr/local/bin/supervisord
    ln -sfn "${SUPERVISORCTL_BIN}" /usr/local/bin/supervisorctl
}

prepare_repository() {
    log "Preparing service user and repository"
    local current_remote deployed_commit
    id "${SERVICE_USER}" >/dev/null 2>&1 || useradd \
        --system --create-home --home-dir "${INSTALL_ROOT}" \
        --shell /usr/sbin/nologin "${SERVICE_USER}"
    install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${INSTALL_ROOT}"

    if [[ ! -d "${REPO_DIR}/.git" ]]; then
        runuser -u "${SERVICE_USER}" -- git clone \
            --branch "${REPO_BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
    else
        chown -R "${SERVICE_USER}:${SERVICE_USER}" "${REPO_DIR}"
        current_remote="$(runuser -u "${SERVICE_USER}" -- \
            git -C "${REPO_DIR}" remote get-url origin)"
        [[ "${current_remote}" == "${REPO_URL}" ]] \
            || die "${REPO_DIR} belongs to another Git remote."
        runuser -u "${SERVICE_USER}" -- \
            git -C "${REPO_DIR}" fetch --prune origin
        runuser -u "${SERVICE_USER}" -- \
            git -C "${REPO_DIR}" checkout --force -B "${REPO_BRANCH}" \
            "origin/${REPO_BRANCH}"
    fi
    deployed_commit="$(runuser -u "${SERVICE_USER}" -- \
        git -C "${REPO_DIR}" rev-parse --short HEAD)"
    log "Using ${REPO_BRANCH} commit ${deployed_commit}"
    cd "${REPO_DIR}"
}

install_venv() {
    local venv_dir="$1" install_spec="$2"
    [[ -x "${venv_dir}/bin/python" ]] \
        || runuser -u "${SERVICE_USER}" -- python3 -m venv "${venv_dir}"
    runuser -u "${SERVICE_USER}" -- \
        "${venv_dir}/bin/python" -m pip install --upgrade pip
    runuser -u "${SERVICE_USER}" -- \
        "${venv_dir}/bin/python" -m pip install -e "${install_spec}"
}

install_application() {
    log "Installing the model-only inference service"
    local api_enabled=false api_spec
    if [[ "${ENABLE_CAPTION}" == "true" \
        || "${ENABLE_OCR}" == "true" \
        || "${ENABLE_VISUAL_EMBEDDING}" == "true" \
        || "${ENABLE_CAPTION_EMBEDDING}" == "true" \
        || "${ENABLE_RERANKER}" == "true" \
        || "${ENABLE_VQA}" == "true" \
        || "${ENABLE_QUERY_PREPARATION}" == "true" ]]; then
        api_enabled=true
    fi

    if [[ "${api_enabled}" == "true" ]]; then
        if [[ "${ENABLE_RERANKER}" == "true" \
            && "${ENABLE_VQA}" == "true" ]]; then
            api_spec=".[embedding,reranking,vqa]"
        elif [[ "${ENABLE_RERANKER}" == "true" ]]; then
            api_spec=".[embedding,reranking]"
        elif [[ "${ENABLE_VQA}" == "true" ]]; then
            api_spec=".[embedding,vqa]"
        elif [[ "${ENABLE_ASR}" == "true" ]]; then
            api_spec=".[embedding,transcripts]"
        else
            api_spec=".[embedding]"
        fi
        install_venv "${API_VENV}" "${api_spec}"
    fi

    if [[ "${ENABLE_ASR}" == "true" ]]; then
        if [[ "${ENABLE_RERANKER}" == "true" \
            || "${ENABLE_VQA}" == "true" ]]; then
            # Do not install transcripts into the API venv: its Transformers
            # 4.x Qwen dependency conflicts with the ASR Transformers 5.x
            # dependency.
            ASR_PYTHON="${ASR_VENV}/bin/python"
            install_venv "${ASR_VENV}" ".[transcripts]"
        elif [[ "${api_enabled}" != "true" ]]; then
            install_venv "${API_VENV}" ".[transcripts]"
            ASR_PYTHON="${API_PYTHON}"
        fi
    fi
    install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" \
        "${INSTALL_ROOT}/cache" "${INSTALL_ROOT}/logs"
}

install_cloudflared() {
    command -v cloudflared >/dev/null 2>&1 && return
    log "Downloading cloudflared"
    local arch package
    arch="$(dpkg --print-architecture)"
    [[ "${arch}" == "amd64" || "${arch}" == "arm64" ]] \
        || die "Unsupported architecture: ${arch}"
    package="$(mktemp --suffix=.deb)"
    curl --fail --location --silent --show-error --output "${package}" \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}.deb"
    dpkg -i "${package}"
    rm -f "${package}"
}

install_services() {
    log "Installing Thunder-compatible supervisor programs"
    install -d -m 0750 -o root -g "${SERVICE_USER}" /etc/cloudflared
    printf '%s\n' "${CLOUDFLARE_TUNNEL_TOKEN}" \
        > /etc/cloudflared/tunnel-token
    chown -R root:"${SERVICE_USER}" /etc/cloudflared
    chmod 0640 /etc/cloudflared/tunnel-token

    tee /etc/supervisor/conf.d/hcmai-llm.conf >/dev/null <<EOF
[program:${TUNNEL_SERVICE}]
command=$(command -v cloudflared) tunnel --no-autoupdate run --token-file /etc/cloudflared/tunnel-token
user=${SERVICE_USER}
autostart=true
autorestart=true
startsecs=3
stopasgroup=true
killasgroup=true
stdout_logfile=${INSTALL_ROOT}/logs/cloudflared.log
redirect_stderr=true
EOF

    MODEL_SERVICES=()
    if [[ "${ENABLE_CAPTION}" == "true" \
        || "${ENABLE_OCR}" == "true" \
        || "${ENABLE_VISUAL_EMBEDDING}" == "true" \
        || "${ENABLE_CAPTION_EMBEDDING}" == "true" \
        || "${ENABLE_RERANKER}" == "true" \
        || "${ENABLE_VQA}" == "true" \
        || "${ENABLE_QUERY_PREPARATION}" == "true" ]]; then
        MODEL_SERVICES+=("${API_SERVICE}")
        tee -a /etc/supervisor/conf.d/hcmai-llm.conf >/dev/null <<EOF

[program:${API_SERVICE}]
command=${API_PYTHON} -m uvicorn llm.server.api:app --host ${API_HOST} --port ${API_PORT} --workers 1 --proxy-headers --forwarded-allow-ips=127.0.0.1
directory=${REPO_DIR}
user=${SERVICE_USER}
autostart=true
autorestart=unexpected
startsecs=5
stopasgroup=true
killasgroup=true
environment=HOME="${INSTALL_ROOT}",PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src",PYTHONDONTWRITEBYTECODE="1",HF_HOME="${INSTALL_ROOT}/cache/huggingface",XDG_CACHE_HOME="${INSTALL_ROOT}/cache",TORCH_HOME="${INSTALL_ROOT}/cache/torch",HCMAI_LLM_CONFIG="${REPO_DIR}/llm/config.yaml",HCMAI_ENABLE_CAPTION="${ENABLE_CAPTION}",HCMAI_ENABLE_OCR="${ENABLE_OCR}",HCMAI_ENABLE_ASR="false",HCMAI_ENABLE_VISUAL_EMBEDDING="${ENABLE_VISUAL_EMBEDDING}",HCMAI_ENABLE_CAPTION_EMBEDDING="${ENABLE_CAPTION_EMBEDDING}",HCMAI_ENABLE_RERANKER="${ENABLE_RERANKER}",HCMAI_ENABLE_VQA="${ENABLE_VQA}",HCMAI_ENABLE_QUERY_PREPARATION="${ENABLE_QUERY_PREPARATION}"
stdout_logfile=${INSTALL_ROOT}/logs/caption-ocr.log
redirect_stderr=true
EOF
    fi

    if [[ "${ENABLE_ASR}" == "true" ]]; then
        MODEL_SERVICES+=("${ASR_SERVICE}")
        tee -a /etc/supervisor/conf.d/hcmai-llm.conf >/dev/null <<EOF

[program:${ASR_SERVICE}]
command=${ASR_PYTHON} -m uvicorn llm.server.api:app --host ${ASR_HOST} --port ${ASR_PORT} --workers 1 --proxy-headers --forwarded-allow-ips=127.0.0.1
directory=${REPO_DIR}
user=${SERVICE_USER}
autostart=true
autorestart=unexpected
startsecs=5
stopasgroup=true
killasgroup=true
environment=HOME="${INSTALL_ROOT}",PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src",PYTHONDONTWRITEBYTECODE="1",HF_HOME="${INSTALL_ROOT}/cache/huggingface",XDG_CACHE_HOME="${INSTALL_ROOT}/cache",TORCH_HOME="${INSTALL_ROOT}/cache/torch",HCMAI_LLM_CONFIG="${REPO_DIR}/llm/config.yaml",HCMAI_ENRICHMENT_CONFIG="${REPO_DIR}/configs/prepare.yaml",HCMAI_ENABLE_CAPTION="false",HCMAI_ENABLE_OCR="false",HCMAI_ENABLE_ASR="true",HCMAI_ENABLE_VISUAL_EMBEDDING="false",HCMAI_ENABLE_CAPTION_EMBEDDING="false",HCMAI_ENABLE_RERANKER="false",HCMAI_ENABLE_VQA="false",HCMAI_ENABLE_QUERY_PREPARATION="false"
stdout_logfile=${INSTALL_ROOT}/logs/asr.log
redirect_stderr=true
EOF
    fi

    if ! "${SUPERVISORCTL_BIN}" \
        -c /etc/supervisor/supervisord.conf pid >/dev/null 2>&1; then
        "${SUPERVISORD_BIN}" -c /etc/supervisor/supervisord.conf
    fi
    "${SUPERVISORCTL_BIN}" -c /etc/supervisor/supervisord.conf reread
    "${SUPERVISORCTL_BIN}" -c /etc/supervisor/supervisord.conf update
    local service
    for service in "${MODEL_SERVICES[@]}"; do
        "${SUPERVISORCTL_BIN}" -c /etc/supervisor/supervisord.conf \
            restart "${service}"
    done
    "${SUPERVISORCTL_BIN}" -c /etc/supervisor/supervisord.conf \
        restart "${TUNNEL_SERVICE}"
}

start_and_verify() {
    log "Waiting for local GPU inference services"
    if [[ "${ENABLE_CAPTION}" == "true" \
        || "${ENABLE_OCR}" == "true" \
        || "${ENABLE_VISUAL_EMBEDDING}" == "true" \
        || "${ENABLE_CAPTION_EMBEDDING}" == "true" \
        || "${ENABLE_RERANKER}" == "true" \
        || "${ENABLE_VQA}" == "true" \
        || "${ENABLE_QUERY_PREPARATION}" == "true" ]]; then
        wait_for_service "API" "${API_HOST}" "${API_PORT}" \
            "${INSTALL_ROOT}/logs/caption-ocr.log"
    fi
    if [[ "${ENABLE_ASR}" == "true" ]]; then
        wait_for_service "ASR" "${ASR_HOST}" "${ASR_PORT}" \
            "${INSTALL_ROOT}/logs/asr.log"
    fi
    "${SUPERVISORCTL_BIN}" -c /etc/supervisor/supervisord.conf \
        status "${TUNNEL_SERVICE}" | grep -q RUNNING \
        || { tail -n 80 "${INSTALL_ROOT}/logs/cloudflared.log" >&2; die "Tunnel failed."; }

    local attempt
    for attempt in {1..20}; do
        public_services_ready && return
        sleep 1
    done
    log "Public DNS/tunnel is still propagating; services are running."
}

wait_for_service() {
    local name="$1" host="$2" port="$3" logfile="$4" attempt
    for attempt in {1..900}; do
        curl --fail --silent "http://${host}:${port}/ready" >/dev/null && return
        sleep 1
    done
    tail -n 80 "${logfile}" >&2
    die "${name} service failed readiness."
}

public_ready() {
    local hostname="$1"
    local options=(--fail --silent)
    if [[ -n "${HCMAI_CF_ACCESS_CLIENT_ID}" ]]; then
        options+=(
            --header "CF-Access-Client-Id: ${HCMAI_CF_ACCESS_CLIENT_ID}"
            --header "CF-Access-Client-Secret: ${HCMAI_CF_ACCESS_CLIENT_SECRET}"
        )
    fi
    curl "${options[@]}" "https://${hostname}/ready" >/dev/null
}

public_services_ready() {
    if [[ "${ENABLE_CAPTION}" == "true" \
        || "${ENABLE_OCR}" == "true" \
        || "${ENABLE_VISUAL_EMBEDDING}" == "true" \
        || "${ENABLE_CAPTION_EMBEDDING}" == "true" \
        || "${ENABLE_RERANKER}" == "true" \
        || "${ENABLE_VQA}" == "true" \
        || "${ENABLE_QUERY_PREPARATION}" == "true" ]]; then
        public_ready "${API_PUBLIC_HOSTNAME}" || return 1
    fi
    if [[ "${ENABLE_ASR}" == "true" ]]; then
        public_ready "${ASR_PUBLIC_HOSTNAME}" || return 1
    fi
}

main() {
    parse_args "$@"
    validate
    install_system_dependencies
    prepare_repository
    install_supervisor
    install_application
    install_cloudflared
    install_services
    start_and_verify
    if [[ "${ENABLE_CAPTION}" == "true" \
        || "${ENABLE_OCR}" == "true" \
        || "${ENABLE_VISUAL_EMBEDDING}" == "true" \
        || "${ENABLE_CAPTION_EMBEDDING}" == "true" \
        || "${ENABLE_RERANKER}" == "true" \
        || "${ENABLE_VQA}" == "true" \
        || "${ENABLE_QUERY_PREPARATION}" == "true" ]]; then
        log "API (caption/OCR/embeddings/reranker/VQA/query-prep): https://${API_PUBLIC_HOSTNAME}/docs"
    fi
    if [[ "${ENABLE_ASR}" == "true" ]]; then
        log "ASR: https://${ASR_PUBLIC_HOSTNAME}/docs"
    fi
    printf 'Status: supervisorctl status\n'
    printf 'Restart: sudo supervisorctl restart'
    printf ' %s' "${MODEL_SERVICES[@]}" "${TUNNEL_SERVICE}"
    printf '\n'
    printf 'Logs: tail -f %s/logs/caption-ocr.log %s/logs/asr.log %s/logs/cloudflared.log\n' \
        "${INSTALL_ROOT}" "${INSTALL_ROOT}" "${INSTALL_ROOT}"
}

main "$@"
