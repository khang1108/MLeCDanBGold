#pragma once

#include <cstdint>
#include <string>

namespace hcmai::keyframes_extraction {

struct RationalValue {
    std::int64_t numerator;
    std::int64_t denominator;
};

struct VideoInput {
    std::string video_id;
    std::string watch_url;
    std::int64_t metadata_length_s;
};

struct ExtractionConfig {
    std::int64_t sample_period_ms = 1000;
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
    std::uint64_t sample_index;
    std::int64_t target_timestamp_ms;
    std::int64_t timestamp_ms;
    std::int64_t frame_idx;
    double avg_fps;
    RationalValue avg_fps_rational;
    std::int64_t pts;
    RationalValue time_base;
    int width;
    int height;
    std::string image_path;
    std::string enrichment_image_path;
    std::uint64_t image_size_bytes;
    std::uint64_t enrichment_image_size_bytes;
};

struct NativeVideoManifest {
    std::string video_id;
    std::string status;
    std::int64_t duration_ms;
    std::uint64_t expected_frame_count;
    std::uint64_t emitted_frame_count;
    double avg_fps;
    RationalValue avg_fps_rational;
    std::string extractor_version;
    std::string config_hash;
    std::string frames_jsonl;
};

}  // namespace hcmai::keyframes_extraction
