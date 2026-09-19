#!/usr/bin/env bash
# ==============================================================================
# serving.sh - Orchestrate HCMAI local serving stack using tmux
#
# Services managed in a single tmux session:
#   1. retrieval   : SigLIP2, FAISS indexes, BM25 retrieval server (:8002)
#   2. backend     : FastAPI application gateway (:8000)
#   3. llm         : Local inference API server (:8100)
#   4. frontend    : React UI dashboard (:3000)
#   5. cf-backend  : Cloudflare Tunnel for Backend
#   6. cf-ui       : Cloudflare Tunnel for UI (Frontend)
#
# Usage:
#   ./serving.sh [start|stop|restart|status|attach|help]
# ==============================================================================

set -Eeuo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SESSION_NAME="${SESSION_NAME:-hcmai}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Python environment
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/aic/bin/python}"

# Service endpoints
RETRIEVAL_HOST="${RETRIEVAL_HOST:-127.0.0.1}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8002}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8100}"
LLM_CONFIG="${HCMAI_LLM_CONFIG:-llm/config.yaml}"

FRONTEND_DIR="${REPO_ROOT}/frontend"

# Cloudflare Tunnel Tokens (Backend & UI)
# Bạn có thể điền trực tiếp token vào biến dưới đây hoặc export từ môi trường
CLOUDFLARED_BACKEND_TOKEN="eyJhIjoiZDY1ZDUwM2E1NWM3YWFhODZjYjI3OWU1NzEzMTkyOWMiLCJ0IjoiMzc2YWFhMWMtYjg0Zi00MWYyLWIxNmItMTUzM2E5Yzk4NThiIiwicyI6IlptSTJZakJpT1dRdE0yUmtNUzAwWXprMUxXRTJNVGt0TnpBM04yVmpZVFJpWkRWaiJ9"
CLOUDFLARED_UI_TOKEN="eyJhIjoiZDY1ZDUwM2E1NWM3YWFhODZjYjI3OWU1NzEzMTkyOWMiLCJ0IjoiMGMzNzNkYTUtNTJkZC00YjhkLWFmY2UtNzc4NTBjZmFmZmQ4IiwicyI6Ik16STBNREV4WldRdFlqWm1OaTAwTlRnd0xXRmxabVF0TVRObU9UWTVaR0U1T1dObSJ9"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
log_info() {
    printf "\033[1;34m[INFO]\033[0m %s\n" "$*"
}

log_success() {
    printf "\033[1;32m[SUCCESS]\033[0m %s\n" "$*"
}

log_warn() {
    printf "\033[1;33m[WARN]\033[0m %s\n" "$*"
}

log_error() {
    printf "\033[1;31m[ERROR]\033[0m %s\n" "$*" >&2
}

check_prerequisites() {
    local missing=0

    if ! command -v tmux >/dev/null 2>&1; then
        log_error "Không tìm thấy lệnh 'tmux'. Vui lòng cài đặt (vd: sudo apt install tmux)."
        missing=1
    fi

    if [[ ! -x "$PYTHON_BIN" ]]; then
        log_error "Không tìm thấy Python tại: $PYTHON_BIN"
        missing=1
    fi

    if ! command -v npm >/dev/null 2>&1; then
        log_warn "Không tìm thấy 'npm' trong PATH. Frontend có thể không khởi động được."
    fi

    if ! command -v cloudflared >/dev/null 2>&1; then
        log_warn "Không tìm thấy 'cloudflared' trong PATH."
    fi

    if [[ "$missing" -ne 0 ]]; then
        exit 1
    fi
}

is_session_running() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# ------------------------------------------------------------------------------
# Core Actions
# ------------------------------------------------------------------------------
start_serving() {
    check_prerequisites

    if is_session_running; then
        log_warn "Tmux session '${SESSION_NAME}' đang chạy."
        log_info "Sử dụng './serving.sh status' để xem thông tin hoặc './serving.sh attach' để vào session."
        exit 0
    fi

    log_info "Khởi tạo tmux session: '${SESSION_NAME}'..."

    # 1. Retrieval server
    log_info "1/6 Cấu hình window: retrieval (port ${RETRIEVAL_PORT})..."
    tmux new-session -d -s "$SESSION_NAME" -n "retrieval" -c "$REPO_ROOT"
    tmux send-keys -t "${SESSION_NAME}:retrieval" \
        "$PYTHON_BIN -m hcmai.retrieval.serving.server --host ${RETRIEVAL_HOST} --port ${RETRIEVAL_PORT}" C-m

    # 2. Backend API gateway
    log_info "2/6 Cấu hình window: backend (port ${BACKEND_PORT})..."
    tmux new-window -t "$SESSION_NAME" -n "backend" -c "$REPO_ROOT"
    tmux send-keys -t "${SESSION_NAME}:backend" \
        "$PYTHON_BIN -m uvicorn hcmai.app:app --host ${BACKEND_HOST} --port ${BACKEND_PORT} --reload" C-m

    # 3. LLM API server
    log_info "3/6 Cấu hình window: llm (port ${LLM_PORT})..."
    tmux new-window -t "$SESSION_NAME" -n "llm" -c "$REPO_ROOT"
    tmux send-keys -t "${SESSION_NAME}:llm" \
        "HCMAI_LLM_CONFIG=${LLM_CONFIG} PYTHONPATH=.:src $PYTHON_BIN -m uvicorn llm.server.api:app --host ${LLM_HOST} --port ${LLM_PORT} --workers 1" C-m

    # 4. Frontend UI
    log_info "4/6 Cấu hình window: frontend (port 3000)..."
    tmux new-window -t "$SESSION_NAME" -n "frontend" -c "$FRONTEND_DIR"
    tmux send-keys -t "${SESSION_NAME}:frontend" "npm start" C-m

    # 5. Cloudflared cho Backend
    log_info "5/6 Cấu hình window: cf-backend..."
    tmux new-window -t "$SESSION_NAME" -n "cf-backend" -c "$REPO_ROOT"
    if [[ -n "$CLOUDFLARED_BACKEND_TOKEN" ]]; then
        tmux send-keys -t "${SESSION_NAME}:cf-backend" "cloudflared tunnel run --token \"$CLOUDFLARED_BACKEND_TOKEN\"" C-m
    else
        log_warn "CLOUDFLARED_BACKEND_TOKEN đang để trống. Sẽ chạy 'cloudflared tunnel run'."
        tmux send-keys -t "${SESSION_NAME}:cf-backend" "cloudflared tunnel run" C-m
    fi

    # 6. Cloudflared cho UI (Frontend)
    log_info "6/6 Cấu hình window: cf-ui..."
    tmux new-window -t "$SESSION_NAME" -n "cf-ui" -c "$REPO_ROOT"
    if [[ -n "$CLOUDFLARED_UI_TOKEN" ]]; then
        tmux send-keys -t "${SESSION_NAME}:cf-ui" "cloudflared tunnel run --token \"$CLOUDFLARED_UI_TOKEN\"" C-m
    else
        log_warn "CLOUDFLARED_UI_TOKEN đang để trống. Sẽ chạy 'cloudflared tunnel run'."
        tmux send-keys -t "${SESSION_NAME}:cf-ui" "cloudflared tunnel run" C-m
    fi

    # Mặc định chọn window backend
    tmux select-window -t "${SESSION_NAME}:backend"

    log_success "Đã khởi động toàn bộ dịch vụ trong tmux session '${SESSION_NAME}'!"
    echo ""
    printf "%-15s | %-20s | %-10s\n" "Dịch vụ" "Window Tmux" "Port / Lệnh"
    printf "%s\n" "--------------------------------------------------------"
    printf "%-15s | %-20s | %-10s\n" "Retrieval" "retrieval (0)" "${RETRIEVAL_PORT}"
    printf "%-15s | %-20s | %-10s\n" "Backend API" "backend (1)" "${BACKEND_PORT}"
    printf "%-15s | %-20s | %-10s\n" "LLM Local" "llm (2)" "${LLM_PORT}"
    printf "%-15s | %-20s | %-10s\n" "Frontend UI" "frontend (3)" "3000"
    printf "%-15s | %-20s | %-10s\n" "CF Backend" "cf-backend (4)" "tunnel run"
    printf "%-15s | %-20s | %-10s\n" "CF UI" "cf-ui (5)" "tunnel run"
    echo ""
    log_info "Để tương tác với các màn hình:"
    log_info "  - Gắn terminal vào session : ./serving.sh attach"
    log_info "  - Kiểm tra trạng thái ports: ./serving.sh status"
    log_info "  - Dừng tất cả dịch vụ      : ./serving.sh stop"
}

stop_serving() {
    if ! is_session_running; then
        log_warn "Tmux session '${SESSION_NAME}' hiện không chạy."
        return 0
    fi

    log_info "Đang tắt tmux session '${SESSION_NAME}'..."
    tmux kill-session -t "$SESSION_NAME"
    log_success "Đã tắt toàn bộ tmux session '${SESSION_NAME}'."
}

restart_serving() {
    log_info "Đang restart lại serving stack..."
    stop_serving
    sleep 1
    start_serving
}

check_port() {
    local host="$1"
    local port="$2"
    local name="$3"

    if curl -sS --connect-timeout 1 "http://${host}:${port}/" >/dev/null 2>&1 || \
       curl -sS --connect-timeout 1 "http://${host}:${port}/health" >/dev/null 2>&1 || \
       (command -v nc >/dev/null 2>&1 && nc -z -w 1 "$host" "$port" 2>/dev/null); then
        printf "  \033[1;32m[ONLINE]\033[0m %-14s (http://%s:%s)\n" "$name" "$host" "$port"
    else
        printf "  \033[1;31m[OFFLINE]\033[0m %-14s (http://%s:%s)\n" "$name" "$host" "$port"
    fi
}

status_serving() {
    echo "========================================================"
    echo " HCMAI Serving Stack Status"
    echo "========================================================"

    if is_session_running; then
        printf "Tmux Session: \033[1;32mRUNNING\033[0m (%s)\n\n" "$SESSION_NAME"
        echo "Danh sách windows:"
        tmux list-windows -t "$SESSION_NAME" -F "  - Window ##{window_index}: #{window_name} (active: #{window_active})"
    else
        printf "Tmux Session: \033[1;31mSTOPPED\033[0m (%s)\n" "$SESSION_NAME"
    fi

    echo ""
    echo "Kiểm tra kết nối các cổng:"
    check_port "$RETRIEVAL_HOST" "$RETRIEVAL_PORT" "Retrieval"
    check_port "$BACKEND_HOST" "$BACKEND_PORT" "Backend"
    check_port "$LLM_HOST" "$LLM_PORT" "LLM"
    check_port "127.0.0.1" "3000" "Frontend"
    echo "========================================================"
}

attach_serving() {
    if ! is_session_running; then
        log_error "Tmux session '${SESSION_NAME}' không chạy. Hãy khởi động bằng './serving.sh start' trước."
        exit 1
    fi

    if [[ -n "${TMUX:-}" ]]; then
        log_warn "Bạn đang ở trong một phiên tmux khác. Đang chuyển client sang session '${SESSION_NAME}'..."
        tmux switch-client -t "$SESSION_NAME"
    else
        tmux attach-session -t "$SESSION_NAME"
    fi
}

usage() {
    cat <<EOF
Sử dụng: $(basename "$0") [COMMAND]

Lệnh điều khiển serving stack cho HCMAI:
  start     (Mặc định) Khởi động tất cả dịch vụ trong tmux session '$SESSION_NAME'
  stop      Dừng và đóng toàn bộ tmux session
  restart   Dừng và khởi động lại toàn bộ dịch vụ
  status    Kiểm tra trạng thái tmux và các cổng dịch vụ
  attach    Mở giao diện tmux để xem trực tiếp log các window
  help      Hiển thị hướng dẫn sử dụng này

Phím tắt điều hướng trong tmux sau khi attach:
  - Chuyển window : Ctrl+b sau đó ấn số window (0: retrieval, 1: backend, 2: llm, 3: frontend, 4: cf-backend, 5: cf-ui)
  - Rời khỏi tmux (vẫn giữ service chạy ngầm): Ctrl+b sau đó ấn phím d
EOF
}

# ------------------------------------------------------------------------------
# CLI Dispatcher
# ------------------------------------------------------------------------------
cmd="${1:-start}"

case "$cmd" in
    start)
        start_serving
        ;;
    stop)
        stop_serving
        ;;
    restart)
        restart_serving
        ;;
    status)
        status_serving
        ;;
    attach)
        attach_serving
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Lệnh không hợp lệ: '$cmd'"
        usage
        exit 1
        ;;
esac
