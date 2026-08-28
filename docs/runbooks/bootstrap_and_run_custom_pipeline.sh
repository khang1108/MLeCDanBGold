#!/usr/bin/env bash
# Bootstrap the native keyframe extractor, fetch organizer video archives via
# curl (bypassing yt-dlp through the existing --source-root fixture path), and
# run the custom raw-1fps pipeline end to end.
#
# Usage: run from the repository root on the target host (e.g. ThunderCompute)
# with the "aic" virtualenv already created (python -m venv aic && aic/bin/python
# -m pip install -e '.[embedding]'), or let this script create it.
#
# Configure via environment variables before invoking, e.g.:
#   LIMIT=10 RUN_ROOT=runs/custom-raw1fps-v1 ./docs/runbooks/bootstrap_and_run_custom_pipeline.sh
#
# LIMIT only bounds how many videos the pipeline stage (step 7) processes; it
# does NOT reduce how many archives get downloaded in step 6. Use ZIP_LIMIT to
# fetch only the first N archives for a cheap smoke test, e.g. ZIP_LIMIT=1.
#
# Only OCR calls the thundercompute inference gateway in this pipeline
# (caption/objects/ASR/diarization/embedding stages already run local models
# in-process). Step 4 starts that gateway on localhost so OCR never leaves
# this host; set SKIP_INFERENCE_SERVER=1 to reuse an already-running gateway.
set -euo pipefail


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-runs/custom-raw1fps-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/custom-raw1fps-v1}"
VERSION="${VERSION:-custom-raw1fps-v1}"
FRAME_STORE_ID="${FRAME_STORE_ID:-custom-raw1fps-v1}"
MEDIA_INFO_DIR="${MEDIA_INFO_DIR:-data/media-info-aic25-b1/media-info}"
MEDIA_INFO_ZIP_URL="${MEDIA_INFO_ZIP_URL:-https://aic-data.ledo.io.vn/media-info-aic25-b1.zip}"
LIMIT="${LIMIT:-}"
ZIP_LIMIT="${ZIP_LIMIT:-}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_INFERENCE_SERVER="${SKIP_INFERENCE_SERVER:-0}"
INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${INFERENCE_PORT:-8100}"
HCMAI_INFERENCE_BASE_URL="${HCMAI_INFERENCE_BASE_URL:-http://${INFERENCE_HOST}:${INFERENCE_PORT}}"
export HCMAI_INFERENCE_BASE_URL

INFERENCE_LOG="$RUN_ROOT/inference_gateway.log"

ZIP_DIR="$RUN_ROOT/raw_zips"
SOURCE_ROOT="$RUN_ROOT/videos_source"
BUILD_DIR="build/keyframes_extraction"
NATIVE_EXECUTABLE="$BUILD_DIR/keyframe_extractor"

URLS=(
  https://aic-data.ledo.io.vn/Videos_L21_a.zip
  https://aic-data.ledo.io.vn/Videos_L22_a.zip
  https://aic-data.ledo.io.vn/Videos_L23_a.zip
  https://aic-data.ledo.io.vn/Videos_L24_a.zip
  https://aic-data.ledo.io.vn/Videos_L25_a.zip
  https://aic-data.ledo.io.vn/Videos_L26_a.zip
  https://aic-data.ledo.io.vn/Videos_L26_b.zip
  https://aic-data.ledo.io.vn/Videos_L26_c.zip
  https://aic-data.ledo.io.vn/Videos_L26_d.zip
  https://aic-data.ledo.io.vn/Videos_L26_e.zip
  https://aic-data.ledo.io.vn/Videos_L27_a.zip
  https://aic-data.ledo.io.vn/Videos_L28_a.zip
  https://aic-data.ledo.io.vn/Videos_L29_a.zip
  https://aic-data.ledo.io.vn/Videos_L30_a.zip
)

# --- 1. System build dependencies for src/hcmai/data/cpp/keyframes_extraction ---
if [[ "$SKIP_APT" != "1" ]]; then
  echo "==> installing C++ build dependencies (sudo required)"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake pkg-config ninja-build \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    libjson-c-dev ffmpeg curl unzip
else
  echo "==> SKIP_APT=1, skipping apt-get install"
fi

# --- 2. Configure and build the native extractor ---
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "==> configuring and building keyframe_extractor"
  cmake -S src/hcmai/data/cpp/keyframes_extraction -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --parallel
  ctest --test-dir "$BUILD_DIR" --output-on-failure
else
  echo "==> SKIP_BUILD=1, skipping native build"
fi

# --- 3. Python environment for the offline pipeline ---
if [[ ! -x aic/bin/python ]]; then
  echo "==> creating aic virtualenv"
  python3 -m venv aic
  aic/bin/python -m pip install -e '.[embedding]'
fi

# --- 4. Start the local OCR inference gateway so OCR never calls the public domain ---
mkdir -p "$RUN_ROOT"
if [[ "$SKIP_INFERENCE_SERVER" != "1" ]]; then
  if curl -fsS "$HCMAI_INFERENCE_BASE_URL/health" >/dev/null 2>&1; then
    echo "==> inference gateway already responding at $HCMAI_INFERENCE_BASE_URL"
  else
    echo "==> starting local OCR inference gateway on $HCMAI_INFERENCE_BASE_URL"
    HCMAI_LLM_CONFIG="thundercompute/config.yaml" \
    HCMAI_ENABLE_CAPTION=false \
    HCMAI_ENABLE_OCR=true \
    HCMAI_ENABLE_ASR=false \
    HCMAI_ENABLE_VISUAL_EMBEDDING=false \
    HCMAI_ENABLE_CAPTION_EMBEDDING=false \
    HCMAI_ENABLE_RERANKER=false \
    PYTHONPATH=.:src nohup aic/bin/python -m uvicorn thundercompute.server.api:app \
      --host "$INFERENCE_HOST" --port "$INFERENCE_PORT" --workers 1 \
      >"$INFERENCE_LOG" 2>&1 &
    disown

    for _ in $(seq 1 60); do
      curl -fsS "$HCMAI_INFERENCE_BASE_URL/ready" >/dev/null 2>&1 && break
      sleep 5
    done
    curl -fsS "$HCMAI_INFERENCE_BASE_URL/ready" >/dev/null 2>&1 \
      || { echo "inference gateway did not become ready; see $INFERENCE_LOG"; exit 1; }
  fi
else
  echo "==> SKIP_INFERENCE_SERVER=1, assuming $HCMAI_INFERENCE_BASE_URL is already reachable"
fi

# --- 5. Fetch organizer media-info metadata if not already present locally ---
if [[ ! -d "$MEDIA_INFO_DIR" || -z "$(find "$MEDIA_INFO_DIR" -maxdepth 1 -name '*.json' -print -quit)" ]]; then
  echo "==> media-info missing, downloading $MEDIA_INFO_ZIP_URL"
  media_info_zip="$(mktemp --suffix=.zip)"
  curl -fL --retry 5 --retry-delay 5 -o "$media_info_zip" "$MEDIA_INFO_ZIP_URL"
  media_info_extract="$(mktemp -d)"
  unzip -q "$media_info_zip" -d "$media_info_extract"
  rm -f "$media_info_zip"

  found_dir="$(find "$media_info_extract" -type f -iname 'L[0-9][0-9]_V*.json' -printf '%h\n' | sort -u | head -n1)"
  [[ -n "$found_dir" ]] || { echo "no organizer media-info JSON found in $MEDIA_INFO_ZIP_URL"; exit 1; }

  mkdir -p "$(dirname "$MEDIA_INFO_DIR")"
  rm -rf "$MEDIA_INFO_DIR"
  mv "$found_dir" "$MEDIA_INFO_DIR"
  rm -rf "$media_info_extract"
fi

# --- 6. Download organizer video archives and flatten into {video_id}.mp4 ---
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  mkdir -p "$ZIP_DIR" "$SOURCE_ROOT"
  URLS_TO_FETCH=("${URLS[@]}")
  [[ -z "$ZIP_LIMIT" ]] || URLS_TO_FETCH=("${URLS[@]:0:$ZIP_LIMIT}")
  for url in "${URLS_TO_FETCH[@]}"; do
    fname="$(basename "$url")"
    zip_path="$ZIP_DIR/$fname"

    echo "==> downloading $fname"
    curl -fL -C - --retry 5 --retry-delay 5 -o "$zip_path" "$url"

    echo "==> extracting $fname"
    extract_dir="$(mktemp -d)"
    unzip -q "$zip_path" -d "$extract_dir"

    find "$extract_dir" -type f \( -iname 'L[0-9][0-9]_V*.mp4' -o -iname 'L[0-9][0-9]_V*.mkv' -o -iname 'L[0-9][0-9]_V*.webm' \) -print0 |
      while IFS= read -r -d '' f; do
        stem="$(basename "${f%.*}")"
        mv "$f" "$SOURCE_ROOT/${stem}.mp4"
      done

    rm -rf "$extract_dir" "$zip_path"
  done

  if [[ -z "$ZIP_LIMIT" ]]; then
    missing=0
    for meta in "$MEDIA_INFO_DIR"/*.json; do
      vid="$(basename "${meta%.json}")"
      [[ -f "$SOURCE_ROOT/$vid.mp4" ]] || { echo "MISSING: $vid"; missing=1; }
    done
    [[ "$missing" -eq 0 ]] || { echo "one or more organizer videos are missing; aborting before pipeline run"; exit 1; }
  else
    echo "==> ZIP_LIMIT=$ZIP_LIMIT set, skipping full coverage check (partial download is expected)"
  fi
else
  echo "==> SKIP_DOWNLOAD=1, skipping video download"
fi

# --- 7. Run the existing pipeline against the local source-root fixture ---
PIPELINE_ARGS=(
  --run-root "$RUN_ROOT"
  --output-root "$OUTPUT_ROOT"
  --version "$VERSION"
  --frame-store-id "$FRAME_STORE_ID"
  --media-info-dir "$MEDIA_INFO_DIR"
  --native-executable "$NATIVE_EXECUTABLE"
  --source-root "$SOURCE_ROOT"
  --yt-dlp-binary yt-dlp
)
[[ -z "$LIMIT" ]] || PIPELINE_ARGS+=(--limit "$LIMIT")

echo "==> running prepare_custom_pipeline.py"
PYTHONPATH=.:src aic/bin/python scripts/prepare_custom_pipeline.py "${PIPELINE_ARGS[@]}"

# --- 8. Reclaim disk only after the pipeline run above succeeded ---
echo "==> pipeline succeeded, removing downloaded source videos"
rm -rf "$SOURCE_ROOT"
