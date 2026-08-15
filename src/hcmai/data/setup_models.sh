#!/usr/bin/env bash

# ============================================================
# HCMAI 2026 - Setup Models Script
#
# Installs/downloads:
#   1. TransNetV2
#   2. EfficientGEBD source code and configs
#
# Usage:
#   chmod +x setup_models.sh
#   ./setup_models.sh
# ============================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"

REPO_URL="https://github.com/Ziwei-Zheng/EfficientGEBD.git"
REPO="/models/EfficientGEBD"

CONFIG_DIR="${REPO}/config-files"
BASELINE_CONFIG="${CONFIG_DIR}/baseline.yaml"

HCMAI_CONFIG="${REPO}/hcmai_resnet50_l234.yaml"

CHECKPOINT_DIR="${REPO}/hcmai_checkpoints"
HCMAI_CHECKPOINT="${CHECKPOINT_DIR}/efficientgebd_r50_l234.pth"

# Keep compatibility with existing HCMAI config
COMPAT_CONFIG="${REPO}/model_config.yaml"
COMPAT_CHECKPOINT="${REPO}/model_best.pth"

# Official Google Drive resources
CONFIG_FOLDER_URL="https://drive.google.com/drive/folders/19cNQaVu3Alxn8VYKJX4FKxl0wEFwCDxI"
CHECKPOINT_URL="https://drive.google.com/file/d/1S4M-xnKpjWFGBimcRYzlEDFhDsWQWF_-/view"

FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

TMP_DIR=""

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

log() {
    echo
    echo "============================================================"
    echo "[HCMAI] $*"
    echo "============================================================"
}

info() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARNING] $*" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

cleanup() {
    if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR}" ]]; then
        rm -rf "${TMP_DIR}"
    fi
}

trap cleanup EXIT

trap '
    echo
    echo "[ERROR] Failed at line ${LINENO}: ${BASH_COMMAND}" >&2
' ERR


# ------------------------------------------------------------
# Detect sudo
# ------------------------------------------------------------

if [[ "${EUID}" -eq 0 ]]; then
    SUDO=""
else
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        SUDO=""
    fi
fi


# ------------------------------------------------------------
# Check basic tools
# ------------------------------------------------------------

log "Checking system dependencies"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || die "Python not found: ${PYTHON_BIN}"

if ! command -v git >/dev/null 2>&1; then
    info "Installing git..."
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y git
fi

if ! command -v unzip >/dev/null 2>&1; then
    info "Installing unzip..."
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y unzip
fi

"${PYTHON_BIN}" --version

info "Python executable:"
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
PY


# ------------------------------------------------------------
# Clone TransNetV2
# ------------------------------------------------------------

log "Preparing TransNetV2 repository"

TRANSNET_REPO="/models/TransNetV2"

if [[ -d "${TRANSNET_REPO}/.git" ]]; then
    info "TransNetV2 already exists: ${TRANSNET_REPO}"
else
    info "TransNetV2 not found. Cloning..."
    
    parent_dir="$(dirname "${TRANSNET_REPO}")"
    if [[ ! -d "${parent_dir}" ]]; then
        ${SUDO} mkdir -p "${parent_dir}"
    fi

    if [[ -w "${parent_dir}" ]]; then
        git clone https://github.com/soCzech/TransNetV2.git "${TRANSNET_REPO}"
    else
        ${SUDO} git clone https://github.com/soCzech/TransNetV2.git "${TRANSNET_REPO}"
        if [[ -n "${USER:-}" ]]; then
            ${SUDO} chown -R "${USER}:${USER}" "${TRANSNET_REPO}" || true
        fi
    fi
fi


# ------------------------------------------------------------
# Clone EfficientGEBD
# ------------------------------------------------------------

log "Preparing EfficientGEBD repository"

if [[ -d "${REPO}/.git" ]]; then
    info "Repository already exists:"
    info "${REPO}"
else
    info "Repository not found. Cloning..."

    parent_dir="$(dirname "${REPO}")"

    if [[ ! -d "${parent_dir}" ]]; then
        ${SUDO} mkdir -p "${parent_dir}"
    fi

    if [[ -w "${parent_dir}" ]]; then
        git clone "${REPO_URL}" "${REPO}"
    else
        ${SUDO} git clone "${REPO_URL}" "${REPO}"

        if [[ -n "${USER:-}" ]]; then
            ${SUDO} chown -R "${USER}:${USER}" "${REPO}" || true
        fi
    fi
fi

[[ -f "${REPO}/modeling/config.py" ]] \
    || die "Missing ${REPO}/modeling/config.py"

[[ -f "${REPO}/modeling/baseline.py" ]] \
    || die "Missing ${REPO}/modeling/baseline.py"

info "EfficientGEBD source OK"


# ------------------------------------------------------------
# Install required Python dependencies
# ------------------------------------------------------------

log "Installing required Python dependencies"

"${PYTHON_BIN}" -m pip install -e ".[preprocessing,s3,transcripts,embedding]" \
    "gdown"


# ------------------------------------------------------------
# Validate Torch environment
# ------------------------------------------------------------

log "Checking existing PyTorch / CUDA environment"

"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print("ERROR: PyTorch cannot be imported:", exc)
    sys.exit(1)

try:
    import torchvision
except Exception as exc:
    print("ERROR: torchvision cannot be imported:", exc)
    sys.exit(1)

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


# ------------------------------------------------------------
# Temporary workspace
# ------------------------------------------------------------

TMP_DIR="$(mktemp -d)"
info "Temporary directory: ${TMP_DIR}"


# ------------------------------------------------------------
# Download official config-files
# ------------------------------------------------------------

log "Preparing official EfficientGEBD configs"

if [[ "${FORCE_DOWNLOAD}" == "1" || ! -f "${BASELINE_CONFIG}" ]]; then
    info "Downloading official config-files..."
    CONFIG_TMP="${TMP_DIR}/configs"
    mkdir -p "${CONFIG_TMP}"

    "${PYTHON_BIN}" -m gdown \
        --folder \
        "${CONFIG_FOLDER_URL}" \
        -O "${CONFIG_TMP}"

    BASELINE_FOUND="$(
        find "${CONFIG_TMP}" \
            -type f \
            -name "baseline.yaml" \
            -print \
            -quit
    )"

    if [[ -z "${BASELINE_FOUND}" ]]; then
        echo
        echo "Downloaded files:"
        find "${CONFIG_TMP}" -maxdepth 4 -type f -print || true
        die "baseline.yaml was not found in downloaded config-files"
    fi

    info "Found baseline config: ${BASELINE_FOUND}"

    rm -rf "${CONFIG_DIR}"
    mkdir -p "${CONFIG_DIR}"

    # Copy the complete directory containing baseline.yaml
    cp -a "$(dirname "${BASELINE_FOUND}")/." "${CONFIG_DIR}/"
else
    info "Config already exists, skipping download:"
    info "${BASELINE_CONFIG}"
fi

[[ -f "${BASELINE_CONFIG}" ]] \
    || die "baseline.yaml is still missing: ${BASELINE_CONFIG}"

info "baseline.yaml OK"


# ------------------------------------------------------------
# Download pretrained checkpoint bundle
# ------------------------------------------------------------

log "Preparing EfficientGEBD pretrained checkpoint"

mkdir -p "${CHECKPOINT_DIR}"

if [[ "${FORCE_DOWNLOAD}" == "1" || ! -f "${HCMAI_CHECKPOINT}" ]]; then
    OUTPUT_ZIP="${TMP_DIR}/efficientgebd_output.zip"
    OUTPUT_EXTRACT="${TMP_DIR}/output"

    info "Downloading official pretrained output.zip..."

    "${PYTHON_BIN}" -m gdown \
        "${CHECKPOINT_URL}" \
        -O "${OUTPUT_ZIP}"

    [[ -s "${OUTPUT_ZIP}" ]] \
        || die "Downloaded output.zip is empty"

    info "Downloaded:"
    ls -lh "${OUTPUT_ZIP}"

    info "Validating ZIP..."
    unzip -tq "${OUTPUT_ZIP}" >/dev/null \
        || die "Downloaded checkpoint file is not a valid ZIP"

    mkdir -p "${OUTPUT_EXTRACT}"
    info "Extracting checkpoint bundle..."
    unzip -q "${OUTPUT_ZIP}" -d "${OUTPUT_EXTRACT}"

    CHECKPOINT_FOUND="$(
        find "${OUTPUT_EXTRACT}" \
            -type f \
            -path "*/x2x3x4_r50_eff/model_best.pth" \
            -print \
            -quit
    )"

    if [[ -z "${CHECKPOINT_FOUND}" ]]; then
        echo
        warn "Could not find x2x3x4_r50_eff/model_best.pth"
        warn "Available model_best.pth files:"
        find "${OUTPUT_EXTRACT}" -type f -name "model_best.pth" -print || true
        die "Required EfficientGEBD ResNet50 L2L3L4 checkpoint not found"
    fi

    info "Found checkpoint: ${CHECKPOINT_FOUND}"
    cp "${CHECKPOINT_FOUND}" "${HCMAI_CHECKPOINT}"
else
    info "Checkpoint already exists, skipping download:"
    info "${HCMAI_CHECKPOINT}"
fi

[[ -s "${HCMAI_CHECKPOINT}" ]] \
    || die "Checkpoint missing or empty: ${HCMAI_CHECKPOINT}"

info "Checkpoint installed:"
ls -lh "${HCMAI_CHECKPOINT}"


# ------------------------------------------------------------
# Generate HCMAI merged config
# ------------------------------------------------------------

log "Generating HCMAI EfficientGEBD config"

"${PYTHON_BIN}" - \
    "${REPO}" \
    "${BASELINE_CONFIG}" \
    "${HCMAI_CONFIG}" <<'PY'

from pathlib import Path
import importlib.util
import sys

repo = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

config_py = repo / "modeling" / "config.py"

if not config_py.is_file():
    raise FileNotFoundError(config_py)
if not baseline_path.is_file():
    raise FileNotFoundError(baseline_path)

spec = importlib.util.spec_from_file_location("efficientgebd_config", config_py)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

cfg = module._C.clone()
cfg.merge_from_file(str(baseline_path))

cfg.MODEL.NAME = "BaseModel"
cfg.MODEL.BACKBONE.NAME = "resnet50"
cfg.MODEL.CAT_PREV = True
cfg.MODEL.FPN_START_IDX = 1
cfg.MODEL.HEAD_CHOICE = [3]
cfg.MODEL.IS_BASIC = False

output_path.write_text(cfg.dump(), encoding="utf-8")

print()
print("Generated config:", output_path)
print("Resolved model configuration:")
print("  MODEL.NAME            =", cfg.MODEL.NAME)
print("  MODEL.BACKBONE.NAME   =", cfg.MODEL.BACKBONE.NAME)
print("  MODEL.CAT_PREV        =", cfg.MODEL.CAT_PREV)
print("  MODEL.FPN_START_IDX   =", cfg.MODEL.FPN_START_IDX)
print("  MODEL.HEAD_CHOICE     =", cfg.MODEL.HEAD_CHOICE)
print("  MODEL.IS_BASIC        =", cfg.MODEL.IS_BASIC)
PY

[[ -f "${HCMAI_CONFIG}" ]] || die "Failed to create HCMAI config"


# ------------------------------------------------------------
# Create compatibility symlinks
# ------------------------------------------------------------

log "Creating HCMAI compatibility paths"

rm -f "${COMPAT_CONFIG}"
rm -f "${COMPAT_CHECKPOINT}"

ln -s "${HCMAI_CONFIG}" "${COMPAT_CONFIG}"
ln -s "${HCMAI_CHECKPOINT}" "${COMPAT_CHECKPOINT}"

info "Created symlinks:"
ls -l "${COMPAT_CONFIG}"
ls -l "${COMPAT_CHECKPOINT}"


# ------------------------------------------------------------
# Filesystem preflight
# ------------------------------------------------------------

log "Running filesystem preflight"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
paths = {
    "repo": Path("${REPO}"),
    "source config.py": Path("${REPO}/modeling/config.py"),
    "source baseline.py": Path("${REPO}/modeling/baseline.py"),
    "official baseline": Path("${BASELINE_CONFIG}"),
    "HCMAI config": Path("${HCMAI_CONFIG}"),
    "HCMAI checkpoint": Path("${HCMAI_CHECKPOINT}"),
    "compat config": Path("${COMPAT_CONFIG}"),
    "compat checkpoint": Path("${COMPAT_CHECKPOINT}"),
}
failed = False
for name, path in paths.items():
    ok = path.is_dir() if name == "repo" else path.is_file()
    print(f"{name:24} {'OK' if ok else 'MISSING':8} {path}")
    if not ok:
        failed = True
if failed:
    raise SystemExit("EfficientGEBD filesystem preflight FAILED")
print("\nFilesystem preflight PASSED")
PY


# ------------------------------------------------------------
# Validate generated model config
# ------------------------------------------------------------

log "Validating resolved EfficientGEBD architecture"

"${PYTHON_BIN}" - \
    "${REPO}" \
    "${HCMAI_CONFIG}" <<'PY'

from pathlib import Path
import importlib.util
import sys

repo = Path(sys.argv[1])
config_path = Path(sys.argv[2])
config_py = repo / "modeling" / "config.py"

spec = importlib.util.spec_from_file_location("efficientgebd_config_validate", config_py)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

cfg = module._C.clone()
cfg.merge_from_file(str(config_path))

expected = {
    "MODEL.NAME": (cfg.MODEL.NAME, "BaseModel"),
    "MODEL.BACKBONE.NAME": (cfg.MODEL.BACKBONE.NAME, "resnet50"),
    "MODEL.CAT_PREV": (cfg.MODEL.CAT_PREV, True),
    "MODEL.FPN_START_IDX": (cfg.MODEL.FPN_START_IDX, 1),
    "MODEL.HEAD_CHOICE": (list(cfg.MODEL.HEAD_CHOICE), [3]),
    "MODEL.IS_BASIC": (cfg.MODEL.IS_BASIC, False),
}

failed = False
for key, (actual, wanted) in expected.items():
    ok = actual == wanted
    print(f"{key:28} {actual!s:20} {'OK' if ok else 'INVALID'}")
    if not ok:
        print(f"    expected: {wanted!r}")
        failed = True

if failed:
    raise SystemExit("EfficientGEBD configuration validation FAILED")
print("\nConfiguration validation PASSED")
PY


# ------------------------------------------------------------
# Validate checkpoint structure
# ------------------------------------------------------------

log "Validating checkpoint"

"${PYTHON_BIN}" - \
    "${HCMAI_CHECKPOINT}" <<'PY'

import sys
import torch

path = sys.argv[1]
print("Loading checkpoint on CPU:", path)

try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

if not isinstance(checkpoint, dict):
    raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")
if "model" not in checkpoint:
    raise KeyError("Checkpoint does not contain 'model' state_dict")

state_dict = checkpoint["model"]
if not isinstance(state_dict, dict):
    raise TypeError("checkpoint['model'] is not a state_dict")

print("Checkpoint keys:", list(checkpoint.keys()))
print("Model parameters:", len(state_dict))
if "epoch" in checkpoint:
    print("Checkpoint epoch:", checkpoint["epoch"])
print("\nCheckpoint validation PASSED")
PY

log "Models setup completed successfully"
