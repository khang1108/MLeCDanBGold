#include "hcmai/keyframes_extraction/config.hpp"
#include "hcmai/keyframes_extraction/jsonl.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <string>

int main() {
    using namespace hcmai::keyframes_extraction;
    using namespace hcmai::keyframes_extraction::test_support;

    const auto root = make_temp_directory("jsonl");
    write_text(
        root / "videos.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://youtube.com/watch?v=a\",\"metadata_length_s\":3}\n"
        "{\"video_id\":\"L01_V002\",\"watch_url\":\"https://youtube.com/watch?v=b\",\"metadata_length_s\":4}\n"
    );

    const auto rows = read_video_manifest(root / "videos.jsonl");
    require_true(rows.size() == 2, "two manifest rows expected");
    require_true(rows[0].video_id == "L01_V001", "manifest order must be stable");
    require_true(rows[1].metadata_length_s == 4, "metadata length must be integral");

    write_text(
        root / "duplicate_id.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://a\",\"metadata_length_s\":1}\n"
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://b\",\"metadata_length_s\":1}\n"
    );
    require_throws([&] { read_video_manifest(root / "duplicate_id.jsonl"); });

    write_text(
        root / "duplicate_url.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://same\",\"metadata_length_s\":1}\n"
        "{\"video_id\":\"L01_V002\",\"watch_url\":\"https://same\",\"metadata_length_s\":1}\n"
    );
    require_throws([&] { read_video_manifest(root / "duplicate_url.jsonl"); });

    write_text(
        root / "missing_url.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":null,\"metadata_length_s\":1}\n"
    );
    require_throws([&] { read_video_manifest(root / "missing_url.jsonl"); });

    write_text(
        root / "blank_line.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://a\",\"metadata_length_s\":1}\n\n"
    );
    require_throws([&] { read_video_manifest(root / "blank_line.jsonl"); });

    write_text(root / "malformed.jsonl", "{\"video_id\":\n");
    require_throws([&] { read_video_manifest(root / "malformed.jsonl"); });

    write_text(
        root / "boolean_length.jsonl",
        "{\"video_id\":\"L01_V001\",\"watch_url\":\"https://a\",\"metadata_length_s\":true}\n"
    );
    require_throws([&] { read_video_manifest(root / "boolean_length.jsonl"); });

    write_text(
        root / "unsafe_id.jsonl",
        "{\"video_id\":\"L01/V001\",\"watch_url\":\"https://a\",\"metadata_length_s\":1}\n"
    );
    require_throws([&] { read_video_manifest(root / "unsafe_id.jsonl"); });

    write_text(
        root / "config.json",
        "{"
        "\"sample_period_ms\":1000,"
        "\"durable_long_edge\":1024,"
        "\"durable_jpeg_quality\":92,"
        "\"enrichment_jpeg_quality\":95,"
        "\"write_enrichment_images\":true,"
        "\"yt_dlp_binary\":\"yt-dlp\","
        "\"yt_dlp_cookies_path\":\"/tmp/youtube.cookies.txt\","
        "\"yt_dlp_js_runtime\":\"node\","
        "\"extractor_version\":\"hcmai-keyframes-extractor/0.1.0\","
        "\"config_hash\":\"sha256:test\""
        "}"
    );
    const auto config = read_extraction_config(root / "config.json");
    require_true(config.sample_period_ms == 1000, "sample period must parse");
    require_true(config.durable_long_edge == 1024, "durable edge must parse");
    require_true(config.durable_jpeg_quality == 92, "durable quality must parse");
    require_true(config.write_enrichment_images, "enrichment image flag must parse");
    require_true(
        config.yt_dlp_cookies_path ==
            std::optional<std::string>("/tmp/youtube.cookies.txt"),
        "cookie file path must parse"
    );
    require_true(
        config.yt_dlp_js_runtime == std::optional<std::string>("node"),
        "JavaScript runtime must parse"
    );
    require_true(config.config_hash == "sha256:test", "config hash must parse");

    const NativeFrameRow frame{
        "L01_V001_raw1fps_000000000",
        "L01_V001",
        0,
        0,
        0,
        0,
        29.97,
        RationalValue{30000, 1001},
        0,
        RationalValue{1, 90000},
        64,
        32,
        "images/\"quoted\".jpg",
        "enrichment_images/000000000.jpg",
        128,
        256,
    };
    write_frame_jsonl(root / "frames.jsonl", {frame});
    const auto round_trip = read_frame_jsonl(root / "frames.jsonl");
    require_true(round_trip.size() == 1, "one frame row must round-trip");
    require_true(
        round_trip[0].image_path == "images/\"quoted\".jpg",
        "JSON writer must escape image paths"
    );
    require_true(round_trip[0].avg_fps_rational.denominator == 1001,
        "FPS rational denominator must round-trip");

    write_text(
        root / "invalid_config.json",
        "{"
        "\"sample_period_ms\":0,"
        "\"durable_long_edge\":1024,"
        "\"durable_jpeg_quality\":92,"
        "\"enrichment_jpeg_quality\":95,"
        "\"write_enrichment_images\":true,"
        "\"yt_dlp_binary\":\"yt-dlp\","
        "\"extractor_version\":\"hcmai-keyframes-extractor/0.1.0\","
        "\"config_hash\":\"sha256:test\""
        "}"
    );
    require_throws([&] { read_extraction_config(root / "invalid_config.json"); });

    return finish_test();
}
