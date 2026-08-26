/**
 * @file test_publication.cpp
 * @brief Verifies guarded native enrichment, publication, and cleanup commands.
 *
 * The fixture creates one local video so every lifecycle operation exercises
 * the production CLI and per-video state boundary without network access.
 */

#include "hcmai/keyframes_extraction/extractor.hpp"
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
 * @brief Creates the local source video used by the publication lifecycle test.
 *
 * @param source_root Directory that receives the fixture source video.
 * @param video_id Canonical video identifier encoded into the source filename.
 * @return Path to the generated local MP4 file.
 * @throws std::runtime_error If FFmpeg cannot generate the fixture video.
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
            "unable to generate synthetic publication source: " + result.stderr_text
        );
    }
    return video_path;
}

/**
 * @brief Writes the strict native configuration used by the fixture extraction.
 *
 * @param config_path Destination JSON configuration path.
 * @return None; writes a complete config with stable test provenance.
 */
void write_publication_config(const std::filesystem::path& config_path) {
    test_support::write_text(
        config_path,
        R"({"sample_period_ms":1000,"durable_long_edge":32,"durable_jpeg_quality":92,"enrichment_jpeg_quality":95,"write_enrichment_images":true,"yt_dlp_binary":"yt-dlp","extractor_version":"hcmai-keyframes-extractor/0.1.0","config_hash":"publication-smoke-config"})"
    );
}

/**
 * @brief Writes one compact, valid enrichment handoff for the fixture bundle.
 *
 * @param handoff_path Destination path within the staging bundle.
 * @param video_id Canonical video identifier represented by the handoff.
 * @return None; writes a handoff with every required specialist artifact key.
 */
void write_handoff(
    const std::filesystem::path& handoff_path,
    const std::string& video_id
) {
    test_support::write_text(
        handoff_path,
        "{\"video_id\":\"" + video_id +
            "\",\"frame_count\":3,\"native_manifest_path\":\"staging/" +
            video_id +
            "/manifest.json\",\"frame_id_digest\":\"fixture-frame-digest\","
            "\"frame_store_id\":\"custom-publication-smoke-v1\","
            "\"config_hash\":\"publication-smoke-config\",\"artifacts\":{"
            "\"caption\":{\"path\":\"enrichment/caption/frames.parquet\","
            "\"status\":\"not_evaluated\"},"
            "\"ocr\":{\"path\":\"enrichment/ocr/frames.parquet\","
            "\"status\":\"not_evaluated\"},"
            "\"objects\":{\"path\":\"enrichment/objects/frames.parquet\","
            "\"status\":\"not_evaluated\"},"
            "\"asr\":{\"path\":\"enrichment/asr/segments.parquet\","
            "\"status\":\"not_evaluated\"}}}\n"
    );
}

/**
 * @brief Executes one native state subcommand through the production executable.
 *
 * @param arguments Arguments following the native executable path.
 * @return Process result with captured standard streams and exit details.
 */
ProcessResult run_state_command(const std::vector<std::string>& arguments) {
    std::vector<std::string> command;
    command.reserve(arguments.size() + 1U);
    command.emplace_back(KEYFRAME_EXTRACTOR_PATH);
    command.insert(command.end(), arguments.begin(), arguments.end());
    return run_process(command);
}

/**
 * @brief Exercises successful, idempotent, rejected, and scoped lifecycle paths.
 *
 * @return Zero when all publication lifecycle invariants hold.
 * @throws std::runtime_error If extraction or any lifecycle assertion fails.
 */
int run_publication_lifecycle() {
    using test_support::make_temp_directory;
    using test_support::require_true;

    const std::string video_id = "L01_V001";
    const std::filesystem::path root = make_temp_directory("publication");
    const std::filesystem::path run_root = root / "run-publication";
    const std::filesystem::path source_root = root / "source-root";
    const std::filesystem::path manifest_path = root / "input" / "videos.jsonl";
    const std::filesystem::path config_path = root / "input" / "config.json";
    const std::filesystem::path state_path = run_root / "state" / (video_id + ".json");
    const std::filesystem::path staging_bundle = run_root / "staging" / video_id;
    const std::filesystem::path handoff_path =
        staging_bundle / "enrichment" / "handoff.json";

    make_synthetic_source(source_root, video_id);
    test_support::write_text(
        manifest_path,
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://youtube.com/watch?v=a\",\"metadata_length_s\":3}\n"
    );
    write_publication_config(config_path);
    const ExtractionSummary summary = extract_manifest(ExtractionRequest{
        manifest_path,
        run_root,
        config_path,
        std::nullopt,
        source_root,
        true,
    });
    require_true(summary.completed == 1, "fixture extraction must complete");
    require_true(
        read_state(state_path).status == VideoStatus::EnrichmentPending,
        "fixture must await enrichment"
    );
    write_handoff(handoff_path, video_id);

    const ProcessResult premature_publication = run_state_command({
        "state",
        "mark-published",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
        "--manifest",
        (staging_bundle / "manifest.json").string(),
    });
    require_true(
        premature_publication.exit_code != 0,
        "publication must reject an enrichment_pending predecessor"
    );
    require_true(
        read_state(state_path).status == VideoStatus::EnrichmentPending,
        "rejected publication must preserve enrichment_pending state"
    );

    const ProcessResult enriched = run_state_command({
        "state",
        "mark-enriched",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
        "--artifacts",
        handoff_path.string(),
    });
    require_true(
        enriched.exit_code == 0 && enriched.signal_number == 0,
        "valid handoff must mark the video enriched"
    );
    require_true(
        read_state(state_path).status == VideoStatus::Enriched,
        "handoff must mark enriched"
    );

    const ProcessResult repeated_enriched = run_state_command({
        "state",
        "mark-enriched",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
        "--artifacts",
        handoff_path.string(),
    });
    require_true(
        repeated_enriched.exit_code == 0 && repeated_enriched.signal_number == 0,
        "the same accepted handoff must be idempotent"
    );

    const ProcessResult published = run_state_command({
        "state",
        "mark-published",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
        "--manifest",
        (staging_bundle / "manifest.json").string(),
    });
    require_true(
        published.exit_code == 0 && published.signal_number == 0,
        "enriched video must publish"
    );
    require_true(
        read_state(state_path).status == VideoStatus::Published,
        "publication must be guarded"
    );
    const std::filesystem::path published_bundle = run_root / "published" / video_id;
    require_true(
        test_support::is_regular_file(
            published_bundle / "images" / "000000000.jpg"
        ),
        "durable images must be published"
    );
    require_true(
        test_support::is_regular_file(published_bundle / "manifest.json"),
        "published manifest must be the final commit marker"
    );
    require_true(
        !test_support::exists(staging_bundle),
        "successful publication must move the staging bundle"
    );

    const ProcessResult repeated_published = run_state_command({
        "state",
        "mark-published",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
        "--manifest",
        (published_bundle / "manifest.json").string(),
    });
    require_true(
        repeated_published.exit_code == 0 && repeated_published.signal_number == 0,
        "the published manifest must support idempotent publication"
    );

    const std::filesystem::path other_source = run_root / "source" / "L01_V002.part";
    const std::filesystem::path other_staging =
        run_root / "staging" / "L01_V002" / "enrichment_images" / "keep.jpg";
    test_support::write_text(other_source, "other-video-source");
    test_support::write_text(other_staging, "other-video-ocr");

    const ProcessResult cleaned = run_state_command({
        "state",
        "cleanup",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
    });
    require_true(
        cleaned.exit_code == 0 && cleaned.signal_number == 0,
        "published video must clean up"
    );
    require_true(
        read_state(state_path).status == VideoStatus::Cleaned,
        "cleanup must persist cleaned state"
    );
    require_true(
        !test_support::exists(run_root / "source" / (video_id + ".part")),
        "source must be removed after cleanup"
    );
    require_true(
        !test_support::exists(staging_bundle / "enrichment_images"),
        "temporary OCR images must be removed with the staging bundle"
    );
    require_true(
        test_support::is_regular_file(
            published_bundle / "images" / "000000000.jpg"
        ),
        "published durable image must remain"
    );
    require_true(
        test_support::is_regular_file(published_bundle / "manifest.json"),
        "published manifest must remain after cleanup"
    );
    require_true(
        test_support::is_regular_file(other_source),
        "cleanup must preserve other sources"
    );
    require_true(
        test_support::is_regular_file(other_staging),
        "cleanup must preserve other staging data"
    );

    const ProcessResult repeated_cleanup = run_state_command({
        "state",
        "cleanup",
        "--run-root",
        run_root.string(),
        "--video-id",
        video_id,
    });
    require_true(
        repeated_cleanup.exit_code == 0 && repeated_cleanup.signal_number == 0,
        "cleanup must be idempotent for an already cleaned video"
    );
    return test_support::finish_test();
}

}  // namespace
}  // namespace hcmai::keyframes_extraction

/**
 * @brief Runs the native guarded-publication lifecycle test executable.
 *
 * @param argc Number of command-line arguments; unused.
 * @param argv Command-line argument values; unused.
 * @return Zero when all lifecycle checks pass.
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    return hcmai::keyframes_extraction::run_publication_lifecycle();
}
