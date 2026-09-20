#!/usr/bin/env bash
set -Eeuo pipefail

# HCMAI GPU launcher for the school Slurm cluster.
#
# Run from login02, preferably inside tmux:
#   ./run_gpu.sh --time 04:00:00
#
# Optional:
#   ./run_gpu.sh --time 08:00:00 --task caption
#   ./run_gpu.sh --time 08:00:00 --reinstall
#
# The script:
#   1) redirects HOME/caches to /media02/lttung10
#   2) requests one GPU with Slurm when launched from login02
#   3) creates/activates conda env "research" (Python 3.11)
#   4) detects the GPU-node NVIDIA driver CUDA capability
#   5) installs a compatible PyTorch stack + project/embedding deps with pip
#   6) starts Uvicorn on :8200
#   7) starts cloudflared after /ready succeeds

# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------

BASE_DIR="${BASE_DIR:-/media02/lttung10}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE_DIR/MLeCDanBGold}"
CONDA_ROOT="${CONDA_ROOT:-$BASE_DIR/miniforge3}"
CONDA_ENV="${CONDA_ENV:-research}"

export HOME="$BASE_DIR"
export XDG_CACHE_HOME="$BASE_DIR/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export CONDA_PKGS_DIRS="$BASE_DIR/.conda/pkgs"
export TMPDIR="$BASE_DIR/.tmp"
export PATH="$BASE_DIR/bin:$CONDA_ROOT/bin:$PATH"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$CONDA_PKGS_DIRS" \
  "$TMPDIR" \
  "$BASE_DIR/bin"

# ---------------------------------------------------------------------------
# Defaults / CLI
# ---------------------------------------------------------------------------

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
HCMAI_GPU_TASK="${HCMAI_GPU_TASK:-caption}"
FORCE_REINSTALL=0
INSIDE_SLURM=0

SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-32G}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8200}"
READY_URL="${READY_URL:-http://127.0.0.1:${PORT}/ready}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$BASE_DIR/bin/cloudflared}"
CLOUDFLARED_TUNNEL="${CLOUDFLARED_TUNNEL:-}"
CLOUDFLARED_TOKEN="${CLOUDFLARED_TOKEN:-eyJhIjoiZDY1ZDUwM2E1NWM3YWFhODZjYjI3OWU1NzEzMTkyOWMiLCJ0IjoiMjIxZjNlZWItMGExNi00ZTk5LTljYTQtYWYxYjgyZjlmYmUwIiwicyI6Ik5tSTROalU0T1dNdE9UazRaUzAwTUdNd0xXRTRZbVV0TW1ZMk1EbGpOMk5rTmpNMiJ9}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_gpu.sh [options]

Options:
  -t, --time <Slurm time>   Slurm time limit. Default: 04:00:00
                            Example: --time 08:00:00
      --task <task>         caption|ocr|objects|asr|all|enrichment
      --reinstall           Re-run pip dependency installation
      --help                Show this help

Environment overrides:
  SLURM_PARTITION=batch
  SLURM_GPUS=1
  SLURM_CPUS=8
  SLURM_MEM=32G
  PROJECT_ROOT=/media02/lttung10/MLeCDanBGold
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--time)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      TIME_LIMIT="$2"
      shift 2
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "Missing value for --task" >&2; exit 2; }
      HCMAI_GPU_TASK="$2"
      shift 2
      ;;
    --reinstall)
      FORCE_REINSTALL=1
      shift
      ;;
    --inside-slurm)
      INSIDE_SLURM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$HCMAI_GPU_TASK" in
  caption|ocr|objects|asr|all|enrichment) ;;
  *)
    echo "HCMAI_GPU_TASK must be caption, ocr, objects, asr, all, or enrichment" >&2
    exit 2
    ;;
esac

export HCMAI_GPU_TASK
export HCMAI_LLM_PROFILE=gpu
export HCMAI_LLM_CONFIG="${HCMAI_LLM_CONFIG:-llm/config.yaml}"
export HCMAI_ENRICHMENT_CONFIG="${HCMAI_ENRICHMENT_CONFIG:-configs/vbs_prepare.yaml}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT:$PROJECT_ROOT/src}"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Slurm: when launched on login02, allocate GPU and re-run this script there.
# ---------------------------------------------------------------------------

if [[ "$INSIDE_SLURM" -eq 0 && -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[slurm] requesting GPU"
  echo "[slurm] partition=$SLURM_PARTITION gpu=$SLURM_GPUS cpu=$SLURM_CPUS mem=$SLURM_MEM time=$TIME_LIMIT"

  CHILD_ARGS=(--inside-slurm --time "$TIME_LIMIT" --task "$HCMAI_GPU_TASK")
  if [[ "$FORCE_REINSTALL" -eq 1 ]]; then
    CHILD_ARGS+=(--reinstall)
  fi

  exec srun \
    --partition="$SLURM_PARTITION" \
    --gres="gpu:$SLURM_GPUS" \
    --cpus-per-task="$SLURM_CPUS" \
    --mem="$SLURM_MEM" \
    --time="$TIME_LIMIT" \
    --job-name=hcmai-gpu \
    --export=ALL \
    bash "$SCRIPT_PATH" "${CHILD_ARGS[@]}"
fi

echo "[slurm] job=${SLURM_JOB_ID:-unknown} node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable on $(hostname). Did Slurm allocate a GPU node?" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Conda / Python environment
# ---------------------------------------------------------------------------

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "[conda] Miniforge not found; installing into $CONDA_ROOT"

  INSTALLER="$BASE_DIR/Miniforge3-Linux-x86_64.sh"
  if [[ ! -f "$INSTALLER" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o "$INSTALLER"
    elif command -v wget >/dev/null 2>&1; then
      wget \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O "$INSTALLER"
    else
      echo "Need curl or wget to install Miniforge." >&2
      exit 1
    fi
  fi

  bash "$INSTALLER" -b -p "$CONDA_ROOT"
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
  echo "[conda] creating env '$CONDA_ENV' with Python 3.11"
  conda create -y -n "$CONDA_ENV" python=3.11 pip
fi

conda activate "$CONDA_ENV"
PYTHON_BIN="$(command -v python)"
PYTHON_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# Torch 2.6/cu124 (needed by the current school driver) has a much safer wheel
# path on Python 3.11 than on the base Python 3.14 environment. If an old
# research env was created with another Python, normalize it in-place.
if [[ "$PYTHON_MINOR" != "3.11" ]]; then
  echo "[conda] env '$CONDA_ENV' uses Python $PYTHON_MINOR; switching it to Python 3.11"
  conda install -y -n "$CONDA_ENV" python=3.11 pip
  conda activate "$CONDA_ENV"
  PYTHON_BIN="$(command -v python)"
fi

echo "[conda] env=$CONDA_ENV"
echo "[conda] python=$PYTHON_BIN"
"$PYTHON_BIN" --version

# ---------------------------------------------------------------------------
# Pick a PyTorch build that the CURRENT GPU-node driver can actually run.
# ---------------------------------------------------------------------------

CUDA_DRIVER_MAX="$(
  nvidia-smi 2>/dev/null |
    sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' |
    head -n1
)"

if [[ -z "$CUDA_DRIVER_MAX" ]]; then
  echo "Unable to determine CUDA compatibility from nvidia-smi." >&2
  nvidia-smi >&2 || true
  exit 1
fi

version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

if version_ge "$CUDA_DRIVER_MAX" "12.6"; then
  TORCH_VERSION="2.8.0"
  TORCHVISION_VERSION="0.23.0"
  TORCHAUDIO_VERSION="2.8.0"
  TORCH_CHANNEL="cu126"
elif version_ge "$CUDA_DRIVER_MAX" "12.4"; then
  # The school's current 12.5-capable driver lands here.
  TORCH_VERSION="2.6.0"
  TORCHVISION_VERSION="0.21.0"
  TORCHAUDIO_VERSION="2.6.0"
  TORCH_CHANNEL="cu124"
elif version_ge "$CUDA_DRIVER_MAX" "12.1"; then
  TORCH_VERSION="2.5.1"
  TORCHVISION_VERSION="0.20.1"
  TORCHAUDIO_VERSION="2.5.1"
  TORCH_CHANNEL="cu121"
elif version_ge "$CUDA_DRIVER_MAX" "11.8"; then
  TORCH_VERSION="2.5.1"
  TORCHVISION_VERSION="0.20.1"
  TORCHAUDIO_VERSION="2.5.1"
  TORCH_CHANNEL="cu118"
else
  echo "NVIDIA driver CUDA compatibility $CUDA_DRIVER_MAX is too old for this launcher." >&2
  exit 1
fi

echo "[cuda] driver supports CUDA <= $CUDA_DRIVER_MAX"
echo "[cuda] selected torch=$TORCH_VERSION torchvision=$TORCHVISION_VERSION torchaudio=$TORCHAUDIO_VERSION ($TORCH_CHANNEL)"

# ---------------------------------------------------------------------------
# Pip dependencies
#
# pyproject.toml currently asks for torch>=2.8 / torchvision>=0.23 in some
# groups. On a CUDA-12.5 driver those wheels are too new, so those three
# packages are deliberately filtered out and replaced by the compatible
# versions selected above.
# ---------------------------------------------------------------------------

PYPROJECT="$PROJECT_ROOT/pyproject.toml"
if [[ ! -f "$PYPROJECT" ]]; then
  echo "pyproject.toml not found at $PYPROJECT" >&2
  exit 1
fi

PYPROJECT_HASH="$(
  "$PYTHON_BIN" - "$PYPROJECT" <<'PY'
from pathlib import Path
import hashlib
import sys

p = Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest()[:16])
PY
)"

DEPS_STAMP="$CONDA_PREFIX/.hcmai-deps-${PYPROJECT_HASH}-${TORCH_VERSION}-${TORCH_CHANNEL}"

if [[ "$FORCE_REINSTALL" -eq 1 || ! -f "$DEPS_STAMP" ]]; then
  echo "[pip] installing/updating compatible dependencies"

  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

  "$PYTHON_BIN" -m pip install --upgrade \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL"

  mapfile -t PROJECT_DEPS < <(
    "$PYTHON_BIN" - "$PYPROJECT" <<'PY'
import re
import sys
import tomllib
from pathlib import Path

data = tomllib.loads(Path(sys.argv[1]).read_text())
project = data.get("project", {})

deps = list(project.get("dependencies", []))
deps += list(project.get("optional-dependencies", {}).get("embedding", []))

# Needed by the Qwen-VL caption server; both are declared by the pipeline extra.
deps += [
    "accelerate>=1.12,<2.0",
    "qwen-vl-utils>=0.0.14,<0.1",
]

# Explicit runtime packages requested for this launcher.
deps += [
    "python-multipart",
    "fastapi",
    "uvicorn",
    "pandas",
    "pydantic-settings",
    "pydantic",
]

skip = {"torch", "torchvision", "torchaudio"}
seen = set()

def pkg_name(req: str) -> str:
    # PEP 508 requirement name before extras/version/url markers.
    return re.split(r"[<>=!~ ;@\[]", req, maxsplit=1)[0].strip().lower().replace("_", "-")

for req in deps:
    name = pkg_name(req)
    if not name or name in skip or name in seen:
        continue
    seen.add(name)
    print(req)
PY
  )

  "$PYTHON_BIN" -m pip install --upgrade "${PROJECT_DEPS[@]}"

  # Install the repository itself without letting pip re-resolve the incompatible
  # torch/torchvision lower bounds from pyproject.toml.
  "$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT" --no-deps

  touch "$DEPS_STAMP"
else
  echo "[pip] dependencies already bootstrapped for this pyproject + CUDA stack"
  echo "[pip] use --reinstall to force dependency setup again"
fi

# Verify that the selected PyTorch can actually initialize the allocated GPU.
"$PYTHON_BIN" - <<'PY'
import torch

print("[torch] version:", torch.__version__)
print("[torch] runtime CUDA:", torch.version.cuda)
print("[torch] cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot initialize CUDA on this allocated node.")

print("[torch] GPU:", torch.cuda.get_device_name(0))
PY

# ---------------------------------------------------------------------------
# cloudflared
# ---------------------------------------------------------------------------

if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "[cloudflared] downloading binary to $CLOUDFLARED_BIN"

  if command -v curl >/dev/null 2>&1; then
    curl -L \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -o "$CLOUDFLARED_BIN"
  elif command -v wget >/dev/null 2>&1; then
    wget \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -O "$CLOUDFLARED_BIN"
  else
    echo "Need curl or wget to download cloudflared." >&2
    exit 1
  fi

  chmod +x "$CLOUDFLARED_BIN"
fi

if [[ -z "$CLOUDFLARED_TOKEN" && -z "$CLOUDFLARED_TUNNEL" ]]; then
  echo "Set CLOUDFLARED_TOKEN or CLOUDFLARED_TUNNEL." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Start API + tunnel on the SAME GPU node.
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"

API_PID=""
TUNNEL_PID=""

cleanup() {
  trap - EXIT INT TERM

  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi

  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi

  [[ -n "$TUNNEL_PID" ]] && wait "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "$API_PID" ]] && wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[api] node=$(hostname)"
echo "[api] task=$HCMAI_GPU_TASK"
echo "[api] starting http://${HOST}:${PORT}"

"$PYTHON_BIN" -m uvicorn llm.server.api:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --proxy-headers &
API_PID=$!

echo "[api] waiting up to $READY_TIMEOUT seconds for $READY_URL"

deadline=$((SECONDS + READY_TIMEOUT))
until "$PYTHON_BIN" - "$READY_URL" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=3) as r:
    if not 200 <= r.status < 300:
        raise SystemExit(1)
PY
do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[api] server exited before becoming ready" >&2
    wait "$API_PID"
    exit $?
  fi

  if (( SECONDS >= deadline )); then
    echo "[api] did not become ready within $READY_TIMEOUT seconds" >&2
    exit 1
  fi

  sleep 1
done

echo "[api] ready"
echo "[cloudflared] starting tunnel on the same node: $(hostname)"

if [[ -n "$CLOUDFLARED_TOKEN" ]]; then
  "$CLOUDFLARED_BIN" tunnel run --token "$CLOUDFLARED_TOKEN" &
else
  "$CLOUDFLARED_BIN" tunnel run "$CLOUDFLARED_TUNNEL" &
fi
TUNNEL_PID=$!

# Exit if either long-running child dies.
while true; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[api] stopped" >&2
    wait "$API_PID"
    exit $?
  fi

  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[cloudflared] stopped" >&2
    wait "$TUNNEL_PID"
    exit $?
  fi

  sleep 2
done
