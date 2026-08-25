/**
 * @file test_extractor_smoke.cpp
 * @brief Verifies one local video completes the native extraction lifecycle.
 *
 * The smoke fixture bypasses network acquisition through source_root while
 * exercising the same extract_manifest library path that the CLI uses.
 */

#include "hcmai/keyframes_extraction/extractor.hpp"
#include "hcmai/keyframes_extraction/jsonl.hpp"
#include "hcmai/keyframes_extraction/process.hpp"
#include "hcmai/keyframes_extraction/state.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Creates the local three-second, two-FPS source video used by the smoke test.
 *
 * @param source_root Existing or creatable directory for the local source fixture.
 * @param video_id Canonical video identifier used in the input manifest.
 * @return Path to the generated `{video_id}.mp4` source file.
 * @throws std::runtime_error If local FFmpeg fixture generation fails.
 */
std::filesystem::path make_synthetic_source(
    const std::filesystem::path& source_root,
    const std::string& video_id
) {
    std::filesystem::create_directories(source_root);
    const std::filesystem::path video_path = source_root / (video_id + ".mp4");
    const ProcessResult result = run_process({
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=80x40:rate=2:duration=3",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        video_path.string(),
    });
    if (result.exit_code != 0 || result.signal_number != 0) {
        throw std::runtime_error(
            "unable to generate synthetic extractor source: " + result.stderr_text
        );
    }
    return video_path;
}

/**
 * @brief Writes the strict JSON configuration consumed by the native extractor.
 *
 * @param config_path Destination configuration JSON path.
 * @return None; writes a complete valid configuration document.
 */
void write_smoke_config(const std::filesystem::path& config_path) {
    test_support::write_text(
        config_path,
        R"({"sample_period_ms":1000,"durable_long_edge":32,"durable_jpeg_quality":92,"enrichment_jpeg_quality":95,"write_enrichment_images":true,"yt_dlp_binary":"yt-dlp","extractor_version":"hcmai-keyframes-extractor/0.1.0","config_hash":"extractor-smoke-config"})"
    );
}

/**
 * @brief Executes the complete local one-video extraction smoke fixture.
 *
 * @return Zero when native source preparation, extraction, and checkpointing pass.
 * @throws std::runtime_error If any smoke assertion fails.
 */
int run_extractor_smoke() {
    using test_support::Dimensions;
    using test_support::make_temp_directory;
    using test_support::read_jpeg_dimensions;
    using test_support::require_true;

    const std::string video_id = "L01_V001";
    const std::filesystem::path root = make_temp_directory("extractor-smoke");
    const std::filesystem::path run_root = root / "run-smoke";
    const std::filesystem::path source_root = root / "source-root";
    const std::filesystem::path manifest_path = root / "input" / "videos.jsonl";
    const std::filesystem::path config_path = root / "input" / "config.json";

    make_synthetic_source(source_root, video_id);
    test_support::write_text(
        manifest_path,
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://youtube.com/watch?v=a\",\"metadata_length_s\":3}\n"
    );
    write_smoke_config(config_path);

    const ExtractionSummary summary = extract_manifest(ExtractionRequest{
        manifest_path,
        run_root,
        config_path,
        std::nullopt,
        source_root,
        false,
    });
    require_true(summary.completed == 1, "one synthetic video must complete");
    require_true(summary.failed == 0, "synthetic video must not fail");
    require_true(summary.skipped == 0, "fresh synthetic video must not skip");
    require_true(summary.pending == 0, "all selected videos must reach a terminal result");
    require_true(summary.emitted_frame_count == 3,
                 "three one-FPS samples must be emitted");

    const std::filesystem::path bundle = run_root / "staging" / video_id;
    const std::vector<NativeFrameRow> rows = read_frame_jsonl(
        bundle / "frames.jsonl"
    );
    require_true(rows.size() == 3, "three-second source must emit targets 0, 1, 2");
    require_true(rows[0].frame_idx == 0, "target zero must use frame index zero");
    require_true(rows[1].frame_idx == 2, "one second at two FPS must use index two");
    require_true(rows[2].frame_idx == 4, "two seconds at two FPS must use index four");
    require_true(rows[0].image_path == "images/000000000.jpg",
                 "durable paths must be bundle-relative");
    require_true(rows[0].enrichment_image_path == "enrichment_images/000000000.jpg",
                 "temporary paths must be bundle-relative");

    const std::filesystem::path durable_path = bundle / "images" / "000000000.jpg";
    const std::filesystem::path enrichment_path =
        bundle / "enrichment_images" / "000000000.jpg";
    require_true(
        test_support::file_size(durable_path) > 0,
        "durable image must exist"
    );
    require_true(
        test_support::file_size(enrichment_path) > 0,
        "enrichment image must exist"
    );
    require_true(read_jpeg_dimensions(durable_path) == Dimensions{32, 16},
                 "durable image must respect configured long edge");
    require_true(read_jpeg_dimensions(enrichment_path) == Dimensions{80, 40},
                 "enrichment image must retain source resolution");
    require_true(
        test_support::is_regular_file(bundle / "manifest.json"),
        "native per-video manifest must be published"
    );

    const VideoState state = read_state(run_root / "state" / (video_id + ".json"));
    require_true(state.status == VideoStatus::EnrichmentPending,
                 "extraction must retain the bundle for enrichment");
    require_true(state.emitted_frame_count == 3,
                 "state must retain emitted frame count");
    require_true(
        test_support::is_regular_file(run_root / "source" / (video_id + ".part")),
        "source video must remain available until enrichment completes"
    );

    const ProcessResult cli_result = run_process({
        KEYFRAME_EXTRACTOR_PATH,
        "extract",
        "--manifest",
        manifest_path.string(),
        "--run-root",
        run_root.string(),
        "--config",
        config_path.string(),
        "--source-root",
        source_root.string(),
        "--video-id",
        video_id,
    });
    require_true(cli_result.exit_code == 0 && cli_result.signal_number == 0,
                 "CLI skip invocation must succeed");
    require_true(
        cli_result.stdout_text ==
            "{\"completed\":0,\"failed\":0,\"skipped\":1,\"pending\":0,\"emitted_frame_count\":0}\n",
        "CLI must emit a complete JSON skip summary"
    );

    return test_support::finish_test();
}

/**
 * @brief Verifies failure continuation and clean per-video retry through the CLI.
 *
 * The first invocation deliberately omits one local source while a second
 * manifest row is valid. The retry uses the same run root after adding the
 * missing source, which proves failed state and stale per-video artifacts do
 * not block later extraction.
 *
 * @return Zero when continued failure handling and targeted retry both pass.
 * @throws std::runtime_error If a state or JSON summary assertion fails.
 */
int run_failure_recovery_smoke() {
    using test_support::make_temp_directory;
    using test_support::require_true;

    const std::string missing_video_id = "L01_V002";
    const std::string valid_video_id = "L01_V003";
    const std::filesystem::path root = make_temp_directory("extractor-retry");
    const std::filesystem::path run_root = root / "run-retry";
    const std::filesystem::path source_root = root / "source-root";
    const std::filesystem::path manifest_path = root / "input" / "videos.jsonl";
    const std::filesystem::path config_path = root / "input" / "config.json";

    make_synthetic_source(source_root, valid_video_id);
    test_support::write_text(
        manifest_path,
        "{\"video_id\":\"L01_V002\",\"watch_url\":\"https://youtube.com/watch?v=b\",\"metadata_length_s\":3}\n"
        "{\"video_id\":\"L01_V003\",\"watch_url\":\"https://youtube.com/watch?v=c\",\"metadata_length_s\":3}\n"
    );
    write_smoke_config(config_path);

    const ProcessResult first_result = run_process({
        KEYFRAME_EXTRACTOR_PATH,
        "extract",
        "--manifest",
        manifest_path.string(),
        "--run-root",
        run_root.string(),
        "--config",
        config_path.string(),
        "--source-root",
        source_root.string(),
    });
    require_true(first_result.exit_code == 2 && first_result.signal_number == 0,
                 "one missing source must produce CLI partial-failure status");
    require_true(
        first_result.stdout_text ==
            "{\"completed\":1,\"failed\":1,\"skipped\":0,\"pending\":0,\"emitted_frame_count\":3}\n",
        "CLI must report continued partial failure and successful later video"
    );

    const VideoState failed_state = read_state(
        run_root / "state" / (missing_video_id + ".json")
    );
    require_true(failed_state.status == VideoStatus::Failed,
                 "missing local source must persist failed state");
    require_true(!failed_state.error.empty(),
                 "failed state must retain a diagnostic");
    require_true(
        read_state(run_root / "state" / (valid_video_id + ".json")).status ==
            VideoStatus::EnrichmentPending,
        "later manifest row must complete after an earlier failure"
    );

    make_synthetic_source(source_root, missing_video_id);
    const ProcessResult retry_result = run_process({
        KEYFRAME_EXTRACTOR_PATH,
        "extract",
        "--manifest",
        manifest_path.string(),
        "--run-root",
        run_root.string(),
        "--config",
        config_path.string(),
        "--source-root",
        source_root.string(),
        "--video-id",
        missing_video_id,
    });
    require_true(retry_result.exit_code == 0 && retry_result.signal_number == 0,
                 "targeted retry must complete after source is added");
    require_true(
        retry_result.stdout_text ==
            "{\"completed\":1,\"failed\":0,\"skipped\":0,\"pending\":0,\"emitted_frame_count\":3}\n",
        "targeted retry must report a fresh complete extraction"
    );
    require_true(
        read_state(run_root / "state" / (missing_video_id + ".json")).status ==
            VideoStatus::EnrichmentPending,
        "successful retry must advance from failed to enrichment_pending"
    );

    return test_support::finish_test();
}

}  // namespace
}  // namespace hcmai::keyframes_extraction

/**
 * @brief Runs the native one-video extraction smoke test executable.
 *
 * @param argc Number of command-line arguments; unused.
 * @param argv Command-line arguments; unused.
 * @return Zero when all one-video extraction assertions pass.
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    hcmai::keyframes_extraction::run_extractor_smoke();
    return hcmai::keyframes_extraction::run_failure_recovery_smoke();
}
