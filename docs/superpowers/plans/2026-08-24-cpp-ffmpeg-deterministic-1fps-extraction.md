
# C++/FFmpeg Deterministic 1-FPS Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable Linux C++17/FFmpeg extractor under src/hcmai/data/cpp/keyframes_extraction/ and a Python validation/materialization boundary that produces a competition-correct custom 1-FPS FrameStore without changing the BTC-native baseline.

**Architecture:** Python converts the 873 media-info JSON files into a deterministic native input manifest and later validates native per-video JSONL bundles before writing Parquet. The C++ executable owns yt-dlp source acquisition, FFmpeg decode/timestamp selection, JPEG encoding, canonical custom identity, per-video state, and atomic native bundle publication. Existing Python enrichment stages consume temporary per-video FrameRecord tables; the final corpus materializer publishes only validated durable images and metadata.

**Tech Stack:** C++17, CMake, pkg-config, FFmpeg libavformat/libavcodec/libavutil/libswscale, json-c, Python 3.11, Pydantic FrameRecord, pandas/pyarrow, pytest, and the existing repository artifact helpers.

**Spec:** docs/superpowers/specs/2026-08-24-cpp-ffmpeg-deterministic-1fps-extraction-design.md

## Global Constraints

- Build the native executable as an isolated CMake project under src/hcmai/data/cpp/keyframes_extraction/; it is owned by the data layer but is not imported as a Python module.
- Require Linux C++17 and resolve FFmpeg and json-c through pkg-config; do not add an Arrow C++ dependency.
- Sample every non-negative target timestamp at one-second intervals starting at 0 ms and emit each target at most once.
- Use AVStream::avg_frame_rate as the FPS authority; reject missing, non-positive, or non-finite FPS.
- Compute the competition coordinate exactly as floor(ceil(avg_fps_of_video) * timestamp_ms / 1000) using the selected frame’s actual timestamp.
- Keep frame_id internal and deterministic as {video_id}_raw1fps_{sample_index:09d}; never deduplicate by (video_id, frame_idx).
- Store the selected actual timestamp in FrameRecord.timestamp_ms; retain target timestamp only as native audit metadata.
- Use durable JPEG images with maximum long edge 1024 and quality 92; use source/high-resolution temporary JPEGs with quality 95 for OCR.
- Write state and native manifests through same-directory temporary files followed by atomic rename; a completed state requires validated metadata and image coverage, not image existence alone.
- Restart an interrupted or failed video from a clean staging directory at video granularity; do not implement mid-GOP resume.
- Keep raw source and temporary OCR images until a validated per-video enrichment handoff has been accepted and the native published transition succeeds.
- Python must recompute the custom frame_idx only as a rejection check; it must never silently replace the native value.
- Existing BTC keyframes, BTC mappings, FrameRecord schema, and active BTC preparation behavior remain unchanged.
- Full-corpus execution is gated on a synthetic-video test, one-video ThunderCompute smoke run, representative pilot, count/timestamp/image/state validation, and storage-growth measurement.

---

## File map

Native source and tests are co-located under the user-approved package path:

~~~text
src/hcmai/data/cpp/keyframes_extraction/
├── CMakeLists.txt
├── include/hcmai/keyframes_extraction/
│   ├── config.hpp
│   ├── extractor.hpp
│   ├── ffmpeg.hpp
│   ├── frame_index.hpp
│   ├── jsonl.hpp
│   ├── process.hpp
│   ├── state.hpp
│   ├── timestamp_sampler.hpp
│   └── types.hpp
├── src/
│   ├── config.cpp
│   ├── extractor.cpp
│   ├── ffmpeg.cpp
│   ├── frame_index.cpp
│   ├── jsonl.cpp
│   ├── main.cpp
│   ├── process.cpp
│   ├── state.cpp
│   └── timestamp_sampler.cpp
└── tests/
    ├── test_extractor_smoke.cpp
    ├── test_ffmpeg.cpp
    ├── test_frame_index.cpp
    ├── test_jsonl.cpp
    ├── test_process.cpp
    ├── test_publication.cpp
    ├── test_state.cpp
    ├── test_support.hpp
    └── test_timestamp_sampler.cpp
~~~

Python owns the input-manifest, validation, enrichment handoff, and Parquet
boundary:

~~~text
src/hcmai/data/ingestion/
├── custom_enrichment.py
├── custom_frames.py
├── custom_manifest.py
├── custom_state.py
└── __init__.py                         # export custom ingestion contracts

scripts/
├── materialize_custom_frames.py
└── prepare_custom_extraction.py

configs/custom-extraction.yaml

tests/data/
├── test_custom_enrichment.py
├── test_custom_frames.py
├── test_custom_manifest.py
└── test_custom_state.py

tests/scripts/test_custom_extraction_cli.py
src/hcmai/data/README.md
src/hcmai/data/WORKFLOW.md
~~~

The active src/hcmai/data/pipeline.py, src/hcmai/common/schemas/frame.py, and
BTC ingestion modules are read-only dependencies for this work. They are not
modified unless a focused compatibility test proves a required change.

## Implementation tasks

### Task 1: Scaffold the isolated native CMake target

**Files:**
- Create: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt
- Create: src/hcmai/data/cpp/keyframes_extraction/src/main.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_support.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_process.cpp

**Interfaces:**
- Consumes: installed Linux C++17 compiler, CMake, FFmpeg development packages, and json-c development package.
- Produces: keyframe_extractor executable, keyframes_core static library target, and CTest registration for native tests.

- [x] **Step 1: Write the build smoke test first**

Create a minimal test helper and register a CTest entry that expects the
executable’s version command to return the pinned version string.

~~~cmake
add_test(NAME keyframe_extractor_version COMMAND keyframe_extractor --version)
set_tests_properties(
    keyframe_extractor_version
    PROPERTIES PASS_REGULAR_EXPRESSION "hcmai-keyframes-extractor/0.1.0"
)
~~~

~~~cpp
// tests/test_support.hpp
#pragma once

#include <stdexcept>
#include <string>

inline void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline int finish_test() {
    return 0;
}
~~~

Extend this helper in the same task with make_temp_directory, write_text,
require_throws, file_size, exists, is_regular_file, and read_jpeg_dimensions so
later native tests use one deterministic temporary-fixture implementation.

- [x] **Step 2: Run the build to verify the scaffold is absent**

Run:

~~~bash
cmake -S src/hcmai/data/cpp/keyframes_extraction -B build/keyframes_extraction -DCMAKE_BUILD_TYPE=Debug
cmake --build build/keyframes_extraction
~~~

Expected: FAIL because the package CMake project and native sources do not
exist.

- [x] **Step 3: Implement the minimal CMake project and version command**

Use imported pkg-config targets and keep all native linkage inside the native
target.

~~~cmake
cmake_minimum_required(VERSION 3.20)
project(hcmai_keyframes_extractor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale
)
pkg_check_modules(JSONC REQUIRED IMPORTED_TARGET json-c)

add_library(keyframes_core STATIC
    src/config.cpp
    src/extractor.cpp
    src/ffmpeg.cpp
    src/frame_index.cpp
    src/jsonl.cpp
    src/process.cpp
    src/state.cpp
    src/timestamp_sampler.cpp
)
target_include_directories(keyframes_core PUBLIC include)
target_link_libraries(keyframes_core PUBLIC PkgConfig::FFMPEG PkgConfig::JSONC)
target_compile_options(keyframes_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(keyframe_extractor src/main.cpp)
target_link_libraries(keyframe_extractor PRIVATE keyframes_core)
target_compile_options(keyframe_extractor PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
enable_testing()
add_executable(test_process tests/test_process.cpp)
target_link_libraries(test_process PRIVATE keyframes_core)
target_compile_options(test_process PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME process_unit COMMAND test_process)
add_test(NAME keyframe_extractor_version COMMAND keyframe_extractor --version)
set_tests_properties(
    keyframe_extractor_version
    PROPERTIES PASS_REGULAR_EXPRESSION "hcmai-keyframes-extractor/0.1.0"
)
~~~

Until later tasks add real implementations, provide compile-safe translation
units for the library sources and make main.cpp handle only this command:

~~~cpp
if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::cout << "hcmai-keyframes-extractor/0.1.0\n";
    return 0;
}
~~~

- [x] **Step 4: Build and run the smoke test**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction --output-on-failure
~~~

Expected: the version and initial process test pass.

- [x] **Step 5: Commit the native scaffold**

~~~bash
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "build: scaffold native keyframe extractor"
~~~

### Task 2: Define native contracts and parse JSONL/configuration

**Files:**
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/types.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/config.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/jsonl.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/config.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/jsonl.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_jsonl.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt

**Interfaces:**
- Consumes: JSONL rows containing video_id, watch_url, and metadata_length_s; JSON configuration produced by Python.
- Produces: validated VideoInput, ExtractionConfig, NativeFrameRow, and NativeVideoManifest values used by later native tasks.

- [x] **Step 1: Write failing parser tests**

Use a temporary directory and test the accepted row shape, deterministic order,
duplicate rejection, missing URL rejection, and configuration range
validation.

~~~cpp
int main() {
    const auto root = make_temp_directory("jsonl");
    write_text(root / "videos.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://youtube.com/watch?v=a\",\"metadata_length_s\":3}\n"
        "{\"video_id\":\"L01_V002\",\"watch_url\":\"https://youtube.com/watch?v=b\",\"metadata_length_s\":4}\n");

    const auto rows = read_video_manifest(root / "videos.jsonl");
    require_true(rows.size() == 2, "two manifest rows expected");
    require_true(rows[0].video_id == "L01_V001", "manifest order must be stable");
    require_true(rows[1].metadata_length_s == 4, "metadata length must be integral");

    write_text(root / "duplicate.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://a\",\"metadata_length_s\":1}\n"
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://b\",\"metadata_length_s\":1}\n");
    require_throws([&] { read_video_manifest(root / "duplicate.jsonl"); });
    return finish_test();
}
~~~

- [x] **Step 2: Run the parser test before implementing the contracts**

~~~bash
cmake --build build/keyframes_extraction --target test_jsonl
ctest --test-dir build/keyframes_extraction -R jsonl_unit --output-on-failure
~~~

Expected: FAIL because the contract headers and parser do not exist.

- [x] **Step 3: Implement the exact native value types and JSON parser**

Define shared row values without exposing FFmpeg-owned pointers.

~~~cpp
struct RationalValue {
    int64_t numerator;
    int64_t denominator;
};

struct VideoInput {
    std::string video_id;
    std::string watch_url;
    int64_t metadata_length_s;
};

struct ExtractionConfig {
    int64_t sample_period_ms = 1000;
    int durable_long_edge = 1024;
    int durable_jpeg_quality = 92;
    int enrichment_jpeg_quality = 95;
    bool write_enrichment_images = true;
    std::string yt_dlp_binary = "yt-dlp";
    std::string extractor_version = "hcmai-keyframes-extractor/0.1.0";
    std::string config_hash;
};

struct NativeFrameRow {
    std::string frame_id;
    std::string video_id;
    uint64_t sample_index;
    int64_t target_timestamp_ms;
    int64_t timestamp_ms;
    int64_t frame_idx;
    double avg_fps;
    RationalValue avg_fps_rational;
    int64_t pts;
    RationalValue time_base;
    int width;
    int height;
    std::string image_path;
    std::string enrichment_image_path;
    uint64_t image_size_bytes;
    uint64_t enrichment_image_size_bytes;
};

struct NativeVideoManifest {
    std::string video_id;
    std::string status;
    int64_t duration_ms;
    uint64_t expected_frame_count;
    uint64_t emitted_frame_count;
    double avg_fps;
    RationalValue avg_fps_rational;
    std::string extractor_version;
    std::string config_hash;
    std::string frames_jsonl;
};
~~~

json-c must reject nulls, booleans where integers are required, negative
lengths, blank IDs, duplicate IDs, and malformed JSON. The writer must emit one
compact JSON object per native frame row and escape paths/URLs through json-c,
never string concatenation.

- [x] **Step 4: Build and run parser tests**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction -R jsonl_unit --output-on-failure
~~~

Expected: PASS for accepted rows and malformed-input cases.

- [x] **Step 5: Commit the native contracts**

~~~bash
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "feat: add native extraction contracts"
~~~

### Task 3: Implement the exact competition coordinate, internal identity, and timestamp sampler

**Files:**
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/frame_index.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/frame_index.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/timestamp_sampler.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/timestamp_sampler.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_frame_index.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_timestamp_sampler.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt

**Interfaces:**
- Consumes: positive average FPS, non-negative timestamps, and monotonic decoded timestamps represented by TimedFrame { ordinal, timestamp_ms }.
- Produces: submission_frame_idx, make_frame_id, and SelectedTarget values.

- [x] **Step 1: Write failing identity tests**

~~~cpp
int main() {
    require_true(submission_frame_idx(25.0, 0) == 0, "zero timestamp");
    require_true(submission_frame_idx(25.0, 1000) == 25, "25 FPS");
    require_true(submission_frame_idx(29.97, 1000) == 30, "29.97 FPS ceiling");
    require_true(submission_frame_idx(30.0, 1500) == 45, "30 FPS fractional second");
    require_true(make_frame_id("L01_V001", 0) == "L01_V001_raw1fps_000000000", "first internal ID");
    require_true(make_frame_id("L01_V001", 12) == "L01_V001_raw1fps_000000012", "zero-padded internal ID");
    require_throws([&] { submission_frame_idx(0.0, 1000); });
    require_throws([&] { submission_frame_idx(25.0, -1); });
    return finish_test();
}
~~~

- [x] **Step 2: Write failing sampler tests**

~~~cpp
int main() {
    TimestampSampler sampler(1000);
    auto first = sampler.push(TimedFrame{0, 0});
    require_true(first.size() == 1, "target zero must emit");
    require_true(first[0].selected_ordinal == 0, "first frame selected for target zero");

    auto second = sampler.push(TimedFrame{1, 400});
    require_true(second.empty(), "target one second is not crossed");

    auto third = sampler.push(TimedFrame{2, 1600});
    require_true(third.size() == 1, "target one second must emit once");
    require_true(third[0].selected_ordinal == 1, "nearest previous frame selected");

    TimestampSampler tie(1000);
    tie.push(TimedFrame{0, 500});
    auto tie_result = tie.push(TimedFrame{1, 1500});
    require_true(tie_result[0].selected_ordinal == 0, "exact tie chooses earlier frame");

    TimestampSampler gap(1000);
    gap.push(TimedFrame{0, 0});
    auto crossed = gap.push(TimedFrame{1, 3500});
    require_true(crossed.size() == 3, "all crossed targets emitted exactly once");
    require_true(crossed[0].target_timestamp_ms == 1000, "first crossed target");
    require_true(crossed[2].target_timestamp_ms == 3000, "last crossed target");
    return finish_test();
}
~~~

- [x] **Step 3: Run both tests before implementation**

~~~bash
cmake --build build/keyframes_extraction --target test_frame_index test_timestamp_sampler
ctest --test-dir build/keyframes_extraction -R 'frame_index_unit|timestamp_sampler_unit' --output-on-failure
~~~

Expected: FAIL because the helper functions and sampler do not exist.

- [x] **Step 4: Implement overflow-checked identity and nearest-frame selection**

Use long double for the formula intermediate and reject a result outside the
non-negative signed 64-bit range.

~~~cpp
int64_t submission_frame_idx(double avg_fps, int64_t timestamp_ms) {
    if (!std::isfinite(avg_fps) || avg_fps <= 0.0 || timestamp_ms < 0) {
        throw std::invalid_argument("invalid FPS or timestamp");
    }
    const long double fps_ceiling = std::ceil(static_cast<long double>(avg_fps));
    const long double value = std::floor(
        fps_ceiling * static_cast<long double>(timestamp_ms) / 1000.0L
    );
    if (value > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("frame_idx exceeds int64");
    }
    return static_cast<int64_t>(value);
}
~~~

make_frame_id must reject blank/unsafe video IDs and use exactly
{video_id}_raw1fps_{sample_index:09d}. It must not inspect or derive the
competition coordinate.

TimestampSampler keeps only the previous decoded timestamp and current
timestamp. For every target crossed by current, select previous when its
distance is less than or equal to current distance; the <= comparison is the
exact-tie-earlier rule.

~~~cpp
std::vector<SelectedTarget> TimestampSampler::push(const TimedFrame& current) {
    if (previous_.has_value() && current.timestamp_ms < previous_->timestamp_ms) {
        throw std::invalid_argument("decoded timestamps must be monotonic");
    }
    std::vector<SelectedTarget> selected;
    while (current.timestamp_ms >= next_target_ms_) {
        const TimedFrame* winner = &current;
        if (previous_.has_value()) {
            const int64_t previous_distance = next_target_ms_ - previous_->timestamp_ms;
            const int64_t current_distance = current.timestamp_ms - next_target_ms_;
            if (previous_distance <= current_distance) {
                winner = &previous_.value();
            }
        }
        selected.push_back(SelectedTarget{
            next_target_ms_ / period_ms_,
            next_target_ms_,
            winner->ordinal,
            winner->timestamp_ms,
        });
        next_target_ms_ += period_ms_;
    }
    previous_ = current;
    return selected;
}
~~~

Reject non-positive sample periods and timestamp overflow while advancing the
next target.

- [x] **Step 5: Run tests and commit**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction -R 'frame_index_unit|timestamp_sampler_unit' --output-on-failure
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "feat: implement deterministic one fps sampling"
~~~

Expected: PASS for 25, 29.97, and 30 FPS formula cases, previous/current
selection, exact ties, large gaps, and no duplicate target indexes.

### Task 4: Add FFmpeg decoding, PTS conversion, and JPEG encoding

**Files:**
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/ffmpeg.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/ffmpeg.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_ffmpeg.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt

**Interfaces:**
- Consumes: one local video path, FFmpeg stream metadata, and ImageVariant { max_long_edge, quality }.
- Produces: RAII-owned decoded frames with actual millisecond timestamps, VideoInfo, and encoded JPEG files with byte counts.

- [x] **Step 1: Write the synthetic decode/encode test**

The test creates a local video through an argv-based FFmpeg invocation and never
contacts YouTube.

~~~cpp
run_process({
    "ffmpeg", "-y", "-f", "lavfi",
    "-i", "testsrc=size=80x40:rate=2:duration=3",
    "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
    video_path.string(),
});
VideoDecoder decoder(video_path);
const VideoInfo info = decoder.info();
require_true(info.avg_fps > 1.9 && info.avg_fps < 2.1, "average FPS must be read from stream");
auto decoded = decoder.next();
require_true(decoded.has_value(), "synthetic video must decode");
require_true(decoded->timestamp_ms >= 0, "best effort timestamp must be converted");
const EncodedImage encoded = encode_jpeg(
    *decoded->image,
    output_path,
    ImageVariant{32, 92}
);
require_true(encoded.bytes > 0, "JPEG must contain bytes");
require_true(read_jpeg_dimensions(output_path) == Dimensions{32, 16}, "long edge must be bounded");
~~~

- [x] **Step 2: Run the FFmpeg test before implementation**

~~~bash
cmake --build build/keyframes_extraction --target test_ffmpeg
ctest --test-dir build/keyframes_extraction -R ffmpeg_unit --output-on-failure
~~~

Expected: FAIL because the decoder and encoder wrappers do not exist.

- [x] **Step 3: Implement RAII FFmpeg lifecycle and timestamp authority**

VideoDecoder must open the format, find stream information, select the first
video stream, copy AVStream::avg_frame_rate, reject invalid FPS, open the codec,
decode with avcodec_send_packet/avcodec_receive_frame, read
best_effort_timestamp, reject AV_NOPTS_VALUE, and rescale with
av_rescale_q_rnd(pts, time_base, {1, 1000},
AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX). Retain the raw FPS numerator and
denominator, time-base numerator and denominator, stream duration, width, and
height.

The decoded frame wrapper must clone/refcount an AVFrame before the next packet
is read. The JPEG encoder must use libswscale, preserve aspect ratio, bound the
durable long edge, retain source dimensions when max_long_edge is zero, and
return the final file byte count. Invalid quality values, zero dimensions,
missing output parents, decoder errors, and encoder errors must throw typed
exceptions containing the video path.

- [x] **Step 4: Run codec tests and inspect image metadata**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction -R 'ffmpeg_unit|timestamp_sampler_unit' --output-on-failure
file /tmp/hcmai-keyframes-test/durable.jpg
~~~

Expected: PASS; the synthetic JPEG is readable and its long edge is 32 pixels.

- [x] **Step 5: Commit the FFmpeg layer**

~~~bash
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "feat: add ffmpeg decode and jpeg encoding"
~~~

### Task 5: Add explicit-argv process execution and atomic per-video state

**Files:**
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/process.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/state.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/process.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/state.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_process.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_state.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt

**Interfaces:**
- Consumes: std::vector<std::string> subprocess argv and a run-root state path.
- Produces: captured exit status/stderr, VideoState, atomic state writes, and validated lifecycle transitions.

- [x] **Step 1: Write failing process and state tests**

The process test must prove shell metacharacters remain one literal argument.

~~~cpp
const ProcessResult result = run_process({
    "/usr/bin/printf", "%s", "literal;$(touch /tmp/should-not-exist)",
});
require_true(result.exit_code == 0, "argv process must succeed");
require_true(result.stdout_text == "literal;$(touch /tmp/should-not-exist)", "argv must not use a shell");
~~~

The state test must cover the forward lifecycle and reject an invalid
transition.

~~~cpp
VideoState state = make_pending_state(
    "run-1", "L01_V001", "https://youtube.com/watch?v=a", "hash-1"
);
save_state_atomic(state_path, state);
state = transition_state(state_path, VideoStatus::Pending, VideoStatus::Downloading);
state = transition_state(state_path, VideoStatus::Downloading, VideoStatus::Extracting);
require_throws([&] {
    transition_state(state_path, VideoStatus::Extracting, VideoStatus::Cleaned);
});
require_true(
    read_state(state_path).status == VideoStatus::Extracting,
    "rejected transition must not mutate state"
);
~~~

- [x] **Step 2: Run process/state tests before implementation**

~~~bash
cmake --build build/keyframes_extraction --target test_process test_state
ctest --test-dir build/keyframes_extraction -R 'process_unit|state_unit' --output-on-failure
~~~

Expected: FAIL because no process or state implementation exists.

- [x] **Step 3: Implement POSIX argv execution without shell interpolation**

Use fork, dup2, execvp, and waitpid. Build the child argv array directly from
the vector; never call system, popen, or a shell. Capture stderr into a bounded
string of 64 KiB and preserve exit code and signal number.

~~~cpp
struct ProcessResult {
    int exit_code;
    int signal_number;
    std::string stdout_text;
    std::string stderr_text;
};

ProcessResult run_process(const std::vector<std::string>& argv);
~~~

The yt-dlp caller will later pass arguments equivalent to:

~~~text
yt-dlp --no-playlist --no-progress --newline
       --format bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best
       --merge-output-format mp4
       --output run_root/source/L01_V001.download.%(ext)s
       https://youtube.com/watch?v=video
~~~

The URL is one argv element and is never concatenated into a command string.

- [x] **Step 4: Implement JSON state and allowed transitions**

Represent these statuses exactly:

~~~cpp
enum class VideoStatus {
    Pending, Downloading, Extracting, Extracted,
    EnrichmentPending, Enriched, Published, Cleaned, Failed,
};
~~~

Serialize these fields:

~~~text
run_id, video_id, watch_url, source_path, extractor_version, config_hash,
status, started_at, updated_at, last_completed_sample_index,
emitted_frame_count, native_manifest_path, enrichment_manifest_path, error
~~~

Write to state/{video_id}.json.tmp, flush and close it, then rename it to the
final state path. A transition checks current status, run ID, video ID,
extractor version, and config hash before changing status. Failed state stores
the bounded process/decoder error and keeps staging for diagnosis; a later
extraction run cleans that video’s staging directory before retrying.

- [x] **Step 5: Run tests and commit**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction -R 'process_unit|state_unit' --output-on-failure
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "feat: add native process and checkpoint state"
~~~

Expected: PASS, with no temporary state file left after successful writes.

### Task 6: Implement one-video extraction orchestration

**Files:**
- Create: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/extractor.hpp
- Create: src/hcmai/data/cpp/keyframes_extraction/src/extractor.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_extractor_smoke.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/src/main.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt

**Interfaces:**
- Consumes: VideoInput, ExtractionConfig, run_root, and an optional local source_root used only by offline tests/smoke runs.
- Produces: staging/{video_id}/frames.jsonl, durable/enrichment images, per-video manifest, enrichment_pending state, and ExtractionSummary.

- [x] **Step 1: Write the end-to-end synthetic-video test**

The test must create a 3-second, 2-FPS source, use source_root to bypass network
download, and invoke the same extraction library used by the CLI.

~~~cpp
const ExtractionSummary summary = extract_manifest(ExtractionRequest{
    manifest_path,
    run_root,
    config_path,
    std::nullopt,
    source_root,
    false,
});
require_true(summary.completed == 1, "one synthetic video must complete");
const auto rows = read_frame_jsonl(run_root / "staging/L01_V001/frames.jsonl");
require_true(rows.size() == 3, "3-second stream must produce targets 0, 1, 2");
require_true(rows[0].frame_idx == 0, "target zero coordinate");
require_true(rows[1].frame_idx == 2, "2 FPS coordinate at one second");
require_true(rows[2].frame_idx == 4, "2 FPS coordinate at two seconds");
require_true(
    read_state(run_root / "state/L01_V001.json").status == VideoStatus::EnrichmentPending,
    "extraction must retain source for enrichment"
);
require_true(
    file_size(run_root / "staging/L01_V001/images/000000000.jpg") > 0,
    "durable image must exist"
);
require_true(
    file_size(run_root / "staging/L01_V001/enrichment_images/000000000.jpg") > 0,
    "temporary high-resolution image must exist"
);
~~~

- [x] **Step 2: Run the smoke test before implementing orchestration**

~~~bash
cmake --build build/keyframes_extraction --target test_extractor_smoke
ctest --test-dir build/keyframes_extraction -R extractor_smoke --output-on-failure
~~~

Expected: FAIL because extract_manifest and the native output layout do not
exist.

- [x] **Step 3: Implement per-video source preparation and state progression**

For each selected manifest row:

1. Create state/{video_id}.json with pending if absent.
2. If status is published or cleaned, validate and skip it.
3. Remove only that video’s stale staging/{video_id} and source download before retry.
4. Transition pending/failed to downloading.
5. When source_root is present, copy source_root/{video_id}.mp4 to source/{video_id}.part; otherwise invoke yt-dlp with explicit argv and atomically rename the downloaded media to that path.
6. Transition to extracting.
7. Decode the first video stream, feed actual timestamps into TimestampSampler, and encode every selection immediately so only previous/current decoded frames remain resident.
8. Write native rows to frames.jsonl.tmp, validate count/IDs/byte sizes, then rename to frames.jsonl.
9. Write manifest.json with stream duration, expected target count, emitted count, FPS rational, extractor version, config hash, and relative paths.
10. Transition extracting to extracted and then enrichment_pending.
11. On any error, write failed with the error and continue to the next video unless fail_fast is true.

The native row paths are bundle-relative:

~~~text
image_path:             images/000000000.jpg
enrichment_image_path:  enrichment_images/000000000.jpg
~~~

The final Python materializer will prefix published/{video_id}/ when it creates
portable FrameRecord.image_path values.

- [x] **Step 4: Add CLI parsing for the extraction command**

Support this command surface:

~~~text
keyframe_extractor extract
  --manifest <run_root>/input/media_manifest.jsonl
  --run-root <run_root>
  --config <run_root>/input/extraction_config.json
  [--video-id <video_id>]
  [--source-root <directory>]
  [--fail-fast]
~~~

Return exit code 0 when all selected videos complete, 2 when one or more
videos fail but processing continued, and 1 for invalid CLI/config input. Print
a JSON summary containing completed, failed, skipped, pending, and emitted-frame
counts.

- [x] **Step 5: Run native extraction tests and commit**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction -R 'extractor_smoke|state_unit|ffmpeg_unit' --output-on-failure
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "feat: implement one video ffmpeg extraction"
~~~

Expected: PASS with three rows, two image variants, atomic native manifest,
enrichment_pending state, and no network dependency.

### Task 7: Implement native enrichment, publication, and cleanup commands

**Files:**
- Modify: src/hcmai/data/cpp/keyframes_extraction/include/hcmai/keyframes_extraction/state.hpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/src/state.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/src/main.cpp
- Create: src/hcmai/data/cpp/keyframes_extraction/tests/test_publication.cpp
- Modify: src/hcmai/data/cpp/keyframes_extraction/CMakeLists.txt

**Interfaces:**
- Consumes: validated compact enrichment/publication handoff JSON and an enrichment_pending native bundle.
- Produces: idempotent mark-enriched, mark-published, and cleanup commands with guarded transitions.

- [ ] **Step 1: Write failing lifecycle tests**

~~~cpp
run_state_command({
    "state", "mark-enriched",
    "--run-root", run_root.string(),
    "--video-id", "L01_V001",
    "--artifacts", handoff_path.string(),
});
require_true(
    read_state(state_path).status == VideoStatus::Enriched,
    "handoff must mark enriched"
);

run_state_command({
    "state", "mark-published",
    "--run-root", run_root.string(),
    "--video-id", "L01_V001",
    "--manifest", publication_manifest.string(),
});
require_true(
    read_state(state_path).status == VideoStatus::Published,
    "publication must be guarded"
);
require_true(
    is_regular_file(run_root / "published/L01_V001/images/000000000.jpg"),
    "durable images must be published"
);

run_state_command({
    "state", "cleanup",
    "--run-root", run_root.string(),
    "--video-id", "L01_V001",
});
require_true(
    !exists(run_root / "source/L01_V001.part"),
    "source must be removed after cleanup"
);
require_true(
    !exists(run_root / "staging/L01_V001/enrichment_images"),
    "temporary OCR images must be removed"
);
require_true(
    is_regular_file(run_root / "published/L01_V001/images/000000000.jpg"),
    "published durable image must remain"
);
~~~

- [ ] **Step 2: Run lifecycle tests before implementation**

~~~bash
cmake --build build/keyframes_extraction --target test_publication
ctest --test-dir build/keyframes_extraction -R publication_unit --output-on-failure
~~~

Expected: FAIL because state command handlers and publication moves do not
exist.

- [ ] **Step 3: Implement mark-enriched validation**

Require the handoff JSON to contain video_id, frame_count,
native_manifest_path, frame_id_digest, frame_store_id, config_hash, and an
artifact-path map for Caption, OCR, Objects, and ASR. Check state is
enrichment_pending, compare video_id and config_hash, compare frame_count with
the native manifest, store the handoff path in state, and atomically move to
enriched. Repeating the command with the same handoff is a no-op; a different
handoff or invalid predecessor is an error.

- [ ] **Step 4: Implement mark-published and cleanup**

mark-published must require enriched, verify native and enrichment manifests,
move the complete staging bundle into published/{video_id}/, write
published/{video_id}/manifest.json as the last bundle commit marker, and then
set state to published. A partial move must restore the previous complete
published bundle.

cleanup must require published, remove only the selected video’s
source/{video_id}.part, staging directory, and temporary download remnants,
then set state to cleaned. It must never remove published durable images, the
native manifest, or another video’s directory.

- [ ] **Step 5: Run lifecycle tests and commit**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction -R 'publication_unit|extractor_smoke' --output-on-failure
git add src/hcmai/data/cpp/keyframes_extraction
git commit -m "feat: add guarded native publication lifecycle"
~~~

Expected: PASS for idempotent same-manifest transitions, invalid predecessor
rejection, cleanup scoping, and durable-image preservation.

### Task 8: Build the deterministic Python input manifest and run configuration

**Files:**
- Create: src/hcmai/data/ingestion/custom_manifest.py
- Create: configs/custom-extraction.yaml
- Create: scripts/prepare_custom_extraction.py
- Create: tests/data/test_custom_manifest.py
- Create: tests/scripts/test_custom_extraction_cli.py
- Modify: src/hcmai/data/ingestion/__init__.py

**Interfaces:**
- Consumes: data/media-info-aic25-b1/media-info/*.json records with filename-stem video IDs, watch_url, and integral length.
- Produces: deterministic input/media_manifest.jsonl, normalized input/extraction_config.json, and Python manifest/config functions.

- [ ] **Step 1: Write failing input-manifest tests**

~~~python
def test_build_native_input_manifest_is_sorted_and_strict(tmp_path: Path) -> None:
    media_info = tmp_path / "media-info"
    media_info.mkdir()
    (media_info / "L01_V002.json").write_text(
        json.dumps({"watch_url": "https://youtube.com/watch?v=b", "length": 4}),
        encoding="utf-8",
    )
    (media_info / "L01_V001.json").write_text(
        json.dumps({"watch_url": "https://youtube.com/watch?v=a", "length": 3}),
        encoding="utf-8",
    )

    output = build_native_input_manifest(media_info, tmp_path / "input.jsonl")

    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"video_id":"L01_V001","watch_url":"https://youtube.com/watch?v=a","metadata_length_s":3}',
        '{"video_id":"L01_V002","watch_url":"https://youtube.com/watch?v=b","metadata_length_s":4}',
    ]


def test_manifest_rejects_duplicate_urls_and_invalid_lengths(tmp_path: Path) -> None:
    media_info = tmp_path / "media-info"
    media_info.mkdir()
    payload = {"watch_url": "https://youtube.com/watch?v=same", "length": 3}
    (media_info / "L01_V001.json").write_text(json.dumps(payload), encoding="utf-8")
    (media_info / "L01_V002.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate watch_url"):
        build_native_input_manifest(media_info, tmp_path / "input.jsonl")
~~~

- [ ] **Step 2: Run the Python tests before implementation**

~~~bash
pytest -q tests/data/test_custom_manifest.py tests/scripts/test_custom_extraction_cli.py
~~~

Expected: FAIL because the custom manifest module and CLI do not exist.

- [ ] **Step 3: Implement strict deterministic manifest generation**

The module must expose
build_native_input_manifest(media_info_dir: str | Path, output_path: str | Path)
-> Path and
write_extraction_config(config_path: str | Path, *, run_root: str | Path,
native_executable: str | Path, frame_store_id: str, yt_dlp_binary: str | Path)
-> Path. The module must read
sorted JSON files, derive video_id from each filename stem, require a non-empty
watch_url, require an integer length >= 0, reject duplicate IDs and URLs, and
write JSONL through a temporary sibling file. The 873-row corpus must report
873 unique IDs/URLs and preserve the 470,428-second metadata total.

The normalized extraction config must contain:

~~~yaml
sample_period_ms: 1000
durable_long_edge: 1024
durable_jpeg_quality: 92
enrichment_jpeg_quality: 95
write_enrichment_images: true
extractor_version: hcmai-keyframes-extractor/0.1.0
~~~

write_extraction_config serializes those values as JSON and adds a SHA-256
config_hash over the canonical JSON payload. Native state records that hash;
later Python handoffs carry the same value.

- [ ] **Step 4: Implement and test the preparation CLI**

Support:

~~~text
python scripts/prepare_custom_extraction.py
  --media-info-dir data/media-info-aic25-b1/media-info
  --run-root runs/custom-raw1fps-v1
  --native-executable build/keyframes_extraction/keyframe_extractor
  --frame-store-id custom-raw1fps-v1
  --yt-dlp-binary yt-dlp
~~~

Print JSON with video_count, unique_url_count, metadata_length_seconds,
sample_period_ms, and both generated paths.

- [ ] **Step 5: Run tests and corpus metadata validation**

~~~bash
pytest -q tests/data/test_custom_manifest.py tests/scripts/test_custom_extraction_cli.py
python scripts/prepare_custom_extraction.py \
  --media-info-dir data/media-info-aic25-b1/media-info \
  --run-root /tmp/hcmai-custom-raw1fps-v1 \
  --native-executable build/keyframes_extraction/keyframe_extractor \
  --frame-store-id custom-raw1fps-v1 \
  --yt-dlp-binary yt-dlp
~~~

Expected: 873 videos, 873 unique URLs, and 470,428 metadata seconds; no video
is downloaded by this preparation command.

- [ ] **Step 6: Commit the Python input boundary**

~~~bash
git add src/hcmai/data/ingestion/custom_manifest.py src/hcmai/data/ingestion/__init__.py configs/custom-extraction.yaml scripts/prepare_custom_extraction.py tests/data/test_custom_manifest.py tests/scripts/test_custom_extraction_cli.py
git commit -m "feat: prepare custom extraction manifest"
~~~

### Task 9: Validate native JSONL bundles and map rows to FrameRecord

**Files:**
- Create: src/hcmai/data/ingestion/custom_frames.py
- Create: tests/data/test_custom_frames.py
- Modify: src/hcmai/data/ingestion/__init__.py

**Interfaces:**
- Consumes: native staging/{video_id}/frames.jsonl or published/{video_id}/frames.jsonl, its per-video manifest, and run_root.
- Produces: NativeValidationReport, validated FrameRecord rows, and rejection on any identity/path/count/formula mismatch.

- [ ] **Step 1: Write failing validation tests**

~~~python
def test_native_rows_map_to_frame_records_without_keyframe_order(tmp_path: Path) -> None:
    bundle = write_valid_native_bundle(tmp_path, video_id="L01_V001", fps=29.97, count=2)

    report = validate_native_video_bundle(bundle, run_root=tmp_path, expected_status="published")

    assert report.frame_count == 2
    assert report.duplicate_submission_coordinate_groups == 0
    records = list(iter_native_frame_records(bundle, run_root=tmp_path))
    assert records[0].frame_id == "L01_V001_raw1fps_000000000"
    assert records[0].keyframe_order is None
    assert records[1].frame_idx == math.floor(math.ceil(29.97) * records[1].timestamp_ms / 1000)


def test_native_validation_rejects_formula_mismatch_but_allows_coordinate_collision(tmp_path: Path) -> None:
    bundle = write_valid_native_bundle(tmp_path, video_id="L01_V001", fps=29.97, count=2)
    rows = read_jsonl(bundle / "frames.jsonl")
    rows[1]["frame_idx"] = rows[0]["frame_idx"]
    write_jsonl(bundle / "frames.jsonl", rows)
    with pytest.raises(ValueError, match="frame_idx formula"):
        validate_native_video_bundle(bundle, run_root=tmp_path, expected_status="published")
~~~

- [ ] **Step 2: Run validation tests before implementation**

~~~bash
pytest -q tests/data/test_custom_frames.py
~~~

Expected: FAIL because no native bundle validator exists.

- [ ] **Step 3: Implement typed native-row validation**

Create NativeValidationReport with fields video_id, frame_count,
expected_frame_count, duplicate_submission_coordinate_groups, frame_id_digest,
and config_hash. Expose
validate_native_video_bundle(bundle_root: str | Path, *, run_root: str | Path,
expected_status: Literal["enrichment_pending", "published"] = "published")
-> NativeValidationReport and
iter_native_frame_records(bundle_root: str | Path, *, run_root: str | Path,
image_variant: Literal["durable", "enrichment"] = "durable")
-> Iterator[FrameRecord]. Validate in this
order:

1. per-video manifest status, video ID, extractor version, config hash, and expected count;
2. JSONL row count and strictly increasing sample_index beginning at zero;
3. exact frame_id format and video_id match;
4. non-negative target/actual timestamps and monotonic actual timestamps;
5. positive rational FPS/time base and finite avg_fps;
6. Python recomputation of floor(ceil(avg_fps) * timestamp_ms / 1000);
7. image paths confined under run_root, regular-file existence, positive byte size, and byte-size equality;
8. duplicate frame_id rejection while allowing duplicate (video_id, frame_idx);
9. SHA-256 digest of ordered frame IDs for the enrichment handoff.

Map each validated row to a FrameRecord with keyframe_order=None,
thumbnail_path=None, no shot/event IDs, native pts/time_base, and
selection_reasons=("custom_raw_1fps",). For published bundles, durable paths
must be relative to run_root as published/{video_id}/images/{filename}. For
staging bundles, use staging/{video_id}/images/{filename}. The temporary OCR
variant must never be written into the final global FrameStore.

- [ ] **Step 4: Run validation and existing frame-store regression tests**

~~~bash
pytest -q tests/data/test_custom_frames.py tests/test_data_loader.py tests/test_frame_assets.py
~~~

Expected: PASS, including keyframe_order=None, duplicate
submission-coordinate acceptance, missing-image rejection, and formula mismatch
rejection.

- [ ] **Step 5: Commit the Python validator**

~~~bash
git add src/hcmai/data/ingestion/custom_frames.py src/hcmai/data/ingestion/__init__.py tests/data/test_custom_frames.py
git commit -m "feat: validate native frame bundles"
~~~

### Task 10: Materialize validated custom bundles into Parquet atomically

**Files:**
- Modify: src/hcmai/data/ingestion/custom_frames.py
- Create: scripts/materialize_custom_frames.py
- Modify: tests/data/test_custom_frames.py
- Modify: tests/scripts/test_custom_extraction_cli.py

**Interfaces:**
- Consumes: a run root containing published per-video bundles and a selected-video list/config.
- Produces: corpus/frames.parquet, corpus/manifest.json, and a CustomFrameStoreConfig/materialize_custom_frame_store API that existing FrameStore can load.

- [ ] **Step 1: Write failing materialization tests**

~~~python
def test_materialize_custom_frame_store_publishes_validated_bundle(tmp_path: Path) -> None:
    write_valid_published_bundle(tmp_path, "L01_V001", count=2)
    write_valid_published_bundle(tmp_path, "L01_V002", count=1)

    output = materialize_custom_frame_store(
        CustomFrameStoreConfig(
            run_root=tmp_path,
            output_root=tmp_path / "corpus",
            frame_store_id="custom-raw1fps-v1",
            selected_video_ids=("L01_V001", "L01_V002"),
        )
    )

    table = pd.read_parquet(output)
    assert table["frame_id"].tolist() == [
        "L01_V001_raw1fps_000000000",
        "L01_V001_raw1fps_000000001",
        "L01_V002_raw1fps_000000000",
    ]
    assert table["keyframe_order"].isna().all()
    assert len(FrameStore(output)) == 3
    manifest = json.loads((tmp_path / "corpus/manifest.json").read_text())
    assert manifest["source"] == "custom_raw_video_1fps"


def test_materialization_refuses_one_missing_published_video(tmp_path: Path) -> None:
    write_valid_published_bundle(tmp_path, "L01_V001", count=1)
    with pytest.raises(ValueError, match="missing validated published bundle"):
        materialize_custom_frame_store(
            CustomFrameStoreConfig(
                run_root=tmp_path,
                output_root=tmp_path / "corpus",
                frame_store_id="custom-raw1fps-v1",
                selected_video_ids=("L01_V001", "L01_V002"),
            )
        )
~~~

- [ ] **Step 2: Run materialization tests before implementation**

~~~bash
pytest -q tests/data/test_custom_frames.py tests/scripts/test_custom_extraction_cli.py
~~~

Expected: FAIL because no custom corpus materializer exists.

- [ ] **Step 3: Implement atomic global publication**

Expose CustomFrameStoreConfig with fields run_root, output_root,
frame_store_id, and selected_video_ids, plus
materialize_custom_frame_store(config) returning the output Parquet path.

Read selected IDs in sorted order, validate every published bundle, concatenate
validated FrameRecord.model_dump(mode="python") rows in
(video_id, sample_index) order, and write a staged Parquet plus staged JSON
manifest. Re-read the staged Parquet through FrameRecord.model_validate and
FrameStore before promoting both files. The manifest must include:

~~~json
{
  "pipeline_version": "custom-raw-video-1fps-v1",
  "source": "custom_raw_video_1fps",
  "frame_store_id": "custom-raw1fps-v1",
  "video_count": 873,
  "frame_count": 470428,
  "sample_period_ms": 1000,
  "submission_coordinate_formula": "floor(ceil(avg_fps) * timestamp_ms / 1000)",
  "resume_enabled": true
}
~~~

Include duplicate submission-coordinate diagnostics, per-video counts, and the
ordered frame-ID digest. Refuse publication if any selected video is absent,
not published, invalid, or has an image outside run_root. Do not modify the
existing BTC output directory.

- [ ] **Step 4: Implement the materialization CLI and run it on fixtures**

Support:

~~~text
python scripts/materialize_custom_frames.py
  --run-root /tmp/hcmai-custom-raw1fps-v1
  --output-root /tmp/hcmai-custom-raw1fps-v1/corpus
  --frame-store-id custom-raw1fps-v1
  [--video-id L01_V001]
~~~

Print the output Parquet path and a JSON count summary. The CLI must not invoke
FFmpeg, yt-dlp, or any model.

- [ ] **Step 5: Run tests and commit the Parquet boundary**

~~~bash
pytest -q tests/data/test_custom_frames.py tests/scripts/test_custom_extraction_cli.py tests/test_data_loader.py
git add src/hcmai/data/ingestion/custom_frames.py scripts/materialize_custom_frames.py tests/data/test_custom_frames.py tests/scripts/test_custom_extraction_cli.py
git commit -m "feat: materialize custom frame store"
~~~

Expected: PASS with atomic staged-file cleanup and a FrameStore that preserves
custom frame_id, actual timestamp, and competition frame_idx values.

### Task 11: Add per-video enrichment input variants and handoff validation

**Files:**
- Create: src/hcmai/data/ingestion/custom_enrichment.py
- Create: tests/data/test_custom_enrichment.py
- Modify: src/hcmai/data/ingestion/custom_frames.py
- Modify: src/hcmai/data/ingestion/__init__.py

**Interfaces:**
- Consumes: a validated staging bundle and existing Caption/OCR/Object/ASR artifact paths.
- Produces: temporary durable/high-resolution frames.parquet inputs and a compact handoff manifest suitable for native mark-enriched.

- [ ] **Step 1: Write failing image-variant and handoff tests**

~~~python
def test_enrichment_variants_preserve_identity_but_switch_image_path(tmp_path: Path) -> None:
    bundle = write_valid_staging_bundle(tmp_path, "L01_V001", count=2)

    durable = materialize_video_enrichment_frames(
        bundle, tmp_path / "durable.parquet", image_variant="durable"
    )
    high_res = materialize_video_enrichment_frames(
        bundle, tmp_path / "ocr.parquet", image_variant="enrichment"
    )

    durable_rows = pd.read_parquet(durable)
    high_res_rows = pd.read_parquet(high_res)
    assert durable_rows[["frame_id", "frame_idx", "timestamp_ms"]].equals(
        high_res_rows[["frame_id", "frame_idx", "timestamp_ms"]]
    )
    assert durable_rows["image_path"].str.contains("/images/").all()
    assert high_res_rows["image_path"].str.contains("/enrichment_images/").all()


def test_handoff_rejects_artifact_frame_identity_mismatch(tmp_path: Path) -> None:
    bundle = write_valid_staging_bundle(tmp_path, "L01_V001", count=2)
    bad_caption = write_caption_artifact(tmp_path, frame_ids=["foreign-frame"])
    with pytest.raises(ValueError, match="caption frame identity"):
        write_enrichment_handoff(
            bundle,
            artifact_paths={"caption": bad_caption},
            output_path=tmp_path / "handoff.json",
            frame_store_id="custom-raw1fps-v1",
        )
~~~

- [ ] **Step 2: Run enrichment-boundary tests before implementation**

~~~bash
pytest -q tests/data/test_custom_enrichment.py
~~~

Expected: FAIL because the per-video materializer and handoff validator do not
exist.

- [ ] **Step 3: Implement per-video durable and OCR tables**

Expose materialize_video_enrichment_frames(bundle_root, output_path,
image_variant) returning the generated Parquet path. Use the native rows and
FrameRecord mapping from Task 9. The durable table points to
    staging/{video_id}/images/{filename} and is used by Caption, Objects, and visual
embedding. The enrichment table points to
    staging/{video_id}/enrichment_images/{filename} and is used by OCR. Both tables have
identical frame_id, video_id, frame_idx, timestamp_ms, dimensions, FPS, PTS,
and time-base fields. The high-resolution table is temporary and is never
passed to the final corpus materializer.

- [ ] **Step 4: Implement the artifact handoff validator**

Expose write_enrichment_handoff(bundle_root, artifact_paths, output_path,
frame_store_id) returning the generated JSON path.

Require artifact keys caption, ocr, objects, and asr; allow an artifact to be
explicitly marked not_evaluated while preserving that status. For frame-native
artifacts, validate exact ordered frame IDs plus video_id/frame_idx/timestamp_ms.
For ASR, validate video ID and timeline coverage without forcing one ASR segment
per frame. Write a handoff with video_id, frame_count, frame_id_digest,
frame_store_id, config_hash, native_manifest_path, and an artifacts map
containing each artifact path and validation status.

Use atomic_write and preserve raw specialist artifacts separately from the
handoff summary.

- [ ] **Step 5: Run tests and commit the enrichment boundary**

~~~bash
pytest -q tests/data/test_custom_enrichment.py tests/data/test_custom_frames.py
git add src/hcmai/data/ingestion/custom_enrichment.py src/hcmai/data/ingestion/custom_frames.py src/hcmai/data/ingestion/__init__.py tests/data/test_custom_enrichment.py
git commit -m "feat: add custom enrichment handoff"
~~~

Expected: PASS for high-resolution OCR inputs, durable-input identity, partial
modality status preservation, and artifact identity mismatch rejection.

### Task 12: Add the Python native-state wrapper and operational configuration

**Files:**
- Create: src/hcmai/data/ingestion/custom_state.py
- Modify: src/hcmai/data/ingestion/__init__.py
- Modify: configs/custom-extraction.yaml
- Create: tests/data/test_custom_state.py
- Modify: src/hcmai/data/README.md
- Modify: src/hcmai/data/WORKFLOW.md

**Interfaces:**
- Consumes: native executable path, run root, video ID, and validated handoff/publication manifests.
- Produces: safe Python calls for native state commands and documented ordering for extraction, enrichment, publication, cleanup, embedding, and indexing.

- [ ] **Step 1: Write failing argv-boundary tests**

~~~python
def test_mark_enriched_invokes_native_command_without_shell(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    mark_video_enriched(
        native_executable=tmp_path / "keyframe_extractor",
        run_root=tmp_path / "run",
        video_id="L01_V001",
        handoff_path=tmp_path / "handoff.json",
    )
    assert calls[0]["shell"] is False
    assert calls[0]["argv"] == [
        str(tmp_path / "keyframe_extractor"),
        "state", "mark-enriched",
        "--run-root", str(tmp_path / "run"),
        "--video-id", "L01_V001",
        "--artifacts", str(tmp_path / "handoff.json"),
    ]
~~~

- [ ] **Step 2: Run the wrapper test before implementation**

~~~bash
pytest -q tests/data/test_custom_state.py
~~~

Expected: FAIL because the wrapper functions do not exist.

- [ ] **Step 3: Implement explicit native state wrappers**

Expose mark_video_enriched(native_executable, run_root, video_id,
handoff_path), mark_video_published(native_executable, run_root, video_id,
manifest_path), and cleanup_video(native_executable, run_root, video_id).

Call subprocess.run(argv, check=True, shell=False, capture_output=True,
text=True). Include stderr in raised errors, reject blank IDs/paths, and never
edit state JSON from Python. Preserve native idempotency and predecessor checks.

- [ ] **Step 4: Document the end-to-end run order and image policy**

Update the data documentation with this concrete flow:

~~~text
prepare_custom_extraction.py
  -> native extract
  -> validate staging frames.jsonl
  -> per-video Caption/Object/visual on durable JPEGs
  -> per-video OCR on temporary high-resolution JPEGs
  -> per-video ASR on retained source video
  -> write_enrichment_handoff
  -> native state mark-enriched
  -> native state mark-published
  -> native cleanup
  -> materialize corpus/frames.parquet
  -> build FrameContext, embeddings, and indexes
~~~

State explicitly that BTC preparation remains the active baseline and the
custom corpus uses a separate frame_store_id and run root.

- [ ] **Step 5: Run wrapper/docs tests and commit**

~~~bash
pytest -q tests/data/test_custom_state.py tests/data/test_custom_enrichment.py
git add src/hcmai/data/ingestion/custom_state.py src/hcmai/data/ingestion/__init__.py configs/custom-extraction.yaml src/hcmai/data/README.md src/hcmai/data/WORKFLOW.md tests/data/test_custom_state.py
git commit -m "docs: document custom extraction lifecycle"
~~~

### Task 13: Validate the complete native/Python smoke path before any corpus run

**Files:**
- Modify: tests/scripts/test_custom_extraction_cli.py
- Modify: tests/data/test_custom_frames.py
- Modify: src/hcmai/data/README.md

**Interfaces:**
- Consumes: the built native executable, a synthetic video, the Python manifest/materializer, and native state commands.
- Produces: a reproducible release gate and a recorded decision point for full-corpus execution.

- [ ] **Step 1: Add the smoke acceptance test**

The test must execute these stages in one temporary run root with concrete
arguments:

~~~python
manifest_path = build_native_input_manifest(
    media_info_dir,
    run_root / "input/media_manifest.jsonl",
)
config_path = write_extraction_config(
    run_root / "input/extraction_config.json",
    run_root=run_root,
    native_executable=native_executable,
    frame_store_id="custom-smoke-v1",
    yt_dlp_binary="yt-dlp",
)
subprocess.run(
    [
        str(native_executable), "extract",
        "--manifest", str(manifest_path),
        "--run-root", str(run_root),
        "--config", str(config_path),
        "--source-root", str(synthetic_source_root),
        "--fail-fast",
    ],
    check=True,
    shell=False,
)
validate_native_video_bundle(
    run_root / "staging/L01_V001",
    run_root=run_root,
    expected_status="enrichment_pending",
)
durable_path = materialize_video_enrichment_frames(
    run_root / "staging/L01_V001",
    run_root / "staging/L01_V001/durable_frames.parquet",
    image_variant="durable",
)
ocr_path = materialize_video_enrichment_frames(
    run_root / "staging/L01_V001",
    run_root / "staging/L01_V001/ocr_frames.parquet",
    image_variant="enrichment",
)
handoff_path = write_enrichment_handoff(
    run_root / "staging/L01_V001",
    artifact_paths=make_validated_artifact_map(durable_path, ocr_path),
    output_path=run_root / "staging/L01_V001/enrichment/handoff.json",
    frame_store_id="custom-smoke-v1",
)
mark_video_enriched(native_executable, run_root, "L01_V001", handoff_path)
mark_video_published(
    native_executable,
    run_root,
    "L01_V001",
    run_root / "staging/L01_V001/manifest.json",
)
cleanup_video(native_executable, run_root, "L01_V001")
materialize_custom_frame_store(
    CustomFrameStoreConfig(
        run_root=run_root,
        output_root=run_root / "corpus",
        frame_store_id="custom-smoke-v1",
        selected_video_ids=("L01_V001",),
    )
)
~~~

The test must provide concrete temporary paths and a synthetic source_root to
each call. Assert three 1-FPS rows for a three-second source, monotonic actual
timestamps, the exact formula, readable durable images, readable temporary OCR
images before cleanup, published durable images after cleanup, no source/staging
temporary files after cleanup, and a loadable final FrameStore.

- [ ] **Step 2: Run the complete local smoke gate**

~~~bash
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction --output-on-failure
pytest -q tests/data/test_custom_manifest.py tests/data/test_custom_frames.py tests/data/test_custom_enrichment.py tests/data/test_custom_state.py tests/scripts/test_custom_extraction_cli.py
python -m compileall -q src/hcmai/data/ingestion scripts
git diff --check
~~~

Expected: all native and Python tests pass; no code path downloads a video during
metadata preparation or synthetic smoke tests.

- [ ] **Step 3: Run a one-video ThunderCompute smoke command**

Build Release on the VM, upload one representative media-info row, and run:

~~~bash
cmake -S src/hcmai/data/cpp/keyframes_extraction -B build/keyframes_extraction -DCMAKE_BUILD_TYPE=Release
cmake --build build/keyframes_extraction --parallel
ctest --test-dir build/keyframes_extraction --output-on-failure
python scripts/prepare_custom_extraction.py --media-info-dir data/media-info-aic25-b1/media-info-one --run-root runs/custom-smoke-v1 --native-executable build/keyframes_extraction/keyframe_extractor --frame-store-id custom-smoke-v1 --yt-dlp-binary yt-dlp
build/keyframes_extraction/keyframe_extractor extract --manifest runs/custom-smoke-v1/input/media_manifest.jsonl --run-root runs/custom-smoke-v1 --config runs/custom-smoke-v1/input/extraction_config.json --fail-fast
~~~

Record download time, decode time, emitted count, durable bytes/frame,
temporary bytes/frame, peak source/staging disk, Caption/OCR/Object/visual
quality diagnostics, and cleanup result. OCR acceptance is usable text/region
recall and confidence diagnostics; visual acceptance is embedding cosine
similarity and hand-checkable retrieval ranking. These are measurements, not
assumptions from JPEG quality settings.

- [ ] **Step 4: Run the representative pilot and record the release decision**

Use a bounded, documented subset before selecting the full 873-video run. The
pilot report must contain selected IDs, commit, extractor/config hashes,
hardware/provider, frame count, failure count, storage growth, per-stage
latency, and representative image/OCR failure cases. Approve full-corpus
execution only if counts/timestamps/images/resume/cleanup and storage remain
within the ThunderCompute disk budget.

- [ ] **Step 5: Commit the smoke gate and operational record**

~~~bash
git add tests/data/test_custom_frames.py tests/scripts/test_custom_extraction_cli.py src/hcmai/data/README.md
git commit -m "test: gate custom extraction before corpus run"
~~~

## Spec coverage review

| Design requirement | Plan tasks |
| --- | --- |
| C++17/CMake package path and FFmpeg linkage | 1, 4 |
| JSONL input and native row contract | 2, 8 |
| Exact frame_idx and internal frame_id | 3, 9 |
| One-second target timestamps, nearest selection, earlier ties | 3, 6 |
| avg_frame_rate, PTS/time-base, actual timestamp | 4, 6, 9 |
| Durable JPEG and temporary OCR image policy | 4, 6, 11, 13 |
| Atomic state and video-granularity restart | 5, 6, 7 |
| enrichment_pending -> enriched -> published -> cleaned | 7, 11, 12, 13 |
| Bounded source/staging storage and cleanup safety | 6, 7, 12, 13 |
| Python validation and canonical Parquet materialization | 9, 10 |
| Specialist evidence handoff without flattening | 11, 12 |
| BTC baseline compatibility | 10, 12 |
| Synthetic, smoke, pilot, and full-run gates | 4, 6, 13 |

## Plan self-review checklist

Before handing this plan to an implementer:

1. Scan the plan for incomplete markers, vague implementation phrases, and unfinished code blocks.
2. Run git diff --check and ensure the plan has no whitespace errors.
3. Verify every path in a task appears in the file map or is an intentional modification of an existing file.
4. Verify the Python function names and argument meanings in Tasks 8–12 remain consistent when referenced by later tasks.
5. Verify the native status names and CLI flags are identical in Tasks 5–7 and the Python wrapper in Task 12.
6. Verify no task changes src/hcmai/data/ingestion/btc.py, BTC mappings, or the FrameRecord schema.
7. Run the focused existing tests listed in Task 13 before starting implementation.
