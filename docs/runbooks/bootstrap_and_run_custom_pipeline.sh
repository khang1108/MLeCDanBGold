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
# LIMIT only bounds how many videos the pipeline stage (step 5) processes; it
# does NOT reduce how many archives get downloaded in step 4. Use ZIP_LIMIT to
# fetch only the first N archives for a cheap smoke test, e.g. ZIP_LIMIT=1.
set -euo pipefail


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-runs/custom-raw1fps-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/custom-raw1fps-v1}"
VERSION="${VERSION:-custom-raw1fps-v1}"
FRAME_STORE_ID="${FRAME_STORE_ID:-custom-raw1fps-v1}"
MEDIA_INFO_DIR="${MEDIA_INFO_DIR:-data/media-info-aic25-b1/media-info}"
LIMIT="${LIMIT:-}"
ZIP_LIMIT="${ZIP_LIMIT:-}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

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

# --- 4. Download organizer video archives and flatten into {video_id}.mp4 ---
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

# --- 5. Run the existing pipeline against the local source-root fixture ---
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

# --- 6. Reclaim disk only after the pipeline run above succeeded ---
echo "==> pipeline succeeded, removing downloaded source videos"
rm -rf "$SOURCE_ROOT"
