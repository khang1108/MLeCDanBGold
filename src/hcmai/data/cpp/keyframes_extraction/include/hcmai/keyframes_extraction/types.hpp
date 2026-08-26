/**
 * @file types.hpp
 * @brief Defines value contracts shared by the native keyframe extractor.
 *
 * These contracts preserve source-video, temporal, and artifact identity.
 * They intentionally contain no parsing, decoding, or filesystem behavior.
 */

#pragma once

#include <cstdint>
#include <string>

namespace hcmai::keyframes_extraction
{

    /**
     * @brief Represents an exact signed rational value.
     *
     * The denominator must be non-zero wherever a consuming contract requires
     * a valid ratio.
     */
    struct RationalValue
    {
        /** @brief Numerator of the rational value. */
        std::int64_t numerator;
        /** @brief Denominator of the rational value. */
        std::int64_t denominator;
    };

    /**
     * @brief Describes one source video accepted by the native extractor.
     */
    struct VideoInput
    {
        /** @brief Stable source-video identity supplied by the input manifest. */
        std::string video_id;
        /** @brief HTTP(S) watch URL used by yt-dlp source acquisition. */
        std::string watch_url;
        /** @brief Organizer metadata duration in whole seconds. */
        std::int64_t metadata_length_s;
    };

    /**
     * @brief Contains tunable settings for one native extraction run.
     */
    struct ExtractionConfig
    {
        /** @brief Target sample period in milliseconds; normally 1000. */
        std::int64_t sample_period_ms = 1000;
        /** @brief Maximum durable JPEG long edge in pixels. */
        int durable_long_edge = 1024;
        /** @brief JPEG quality for durable retrieval images. */
        int durable_jpeg_quality = 92;
        /** @brief JPEG quality for temporary enrichment images. */
        int enrichment_jpeg_quality = 95;
        /** @brief Whether temporary high-quality enrichment images are emitted. */
        bool write_enrichment_images = true;
        /** @brief Explicit executable used for source-video acquisition. */
        std::string yt_dlp_binary = "yt-dlp";
        /** @brief Version recorded in native state and manifests. */
        std::string extractor_version = "hcmai-keyframes-extractor/0.1.0";
        /** @brief Deterministic hash identifying the active configuration. */
        std::string config_hash;
    };

    /**
     * @brief Records one durable custom frame and its extraction provenance.
     *
     * frame_id is internal; frame_idx is the competition-facing submission
     * coordinate calculated from the selected actual timestamp.
     */
    struct NativeFrameRow
    {
        /** @brief Deterministic internal frame identity. */
        std::string frame_id;
        /** @brief Source video identity retained for joins and submission. */
        std::string video_id;
        /** @brief Zero-based target ordinal within the custom one-FPS timeline. */
        std::uint64_t sample_index;
        /** @brief Requested sampling target timestamp in milliseconds. */
        std::int64_t target_timestamp_ms;
        /** @brief Actual timestamp of the decoded frame selected for the target. */
        std::int64_t timestamp_ms;
        /** @brief Competition-facing frame coordinate derived from timestamp_ms. */
        std::int64_t frame_idx;
        /** @brief Numeric average FPS from the video stream. */
        double avg_fps;
        /** @brief Exact average FPS rational reported by FFmpeg. */
        RationalValue avg_fps_rational;
        /** @brief FFmpeg presentation timestamp of the selected frame. */
        std::int64_t pts;
        /** @brief FFmpeg time base used to interpret pts. */
        RationalValue time_base;
        /** @brief Encoded durable image width in pixels. */
        int width;
        /** @brief Encoded durable image height in pixels. */
        int height;
        /** @brief Durable JPEG path relative to its native bundle root. */
        std::string image_path;
        /** @brief Optional temporary high-quality enrichment JPEG path. */
        std::string enrichment_image_path;
        /** @brief Byte size of the durable JPEG image. */
        std::uint64_t image_size_bytes;
        /** @brief Byte size of the optional enrichment JPEG image. */
        std::uint64_t enrichment_image_size_bytes;
    };

    /**
     * @brief Summarizes a published or failed native per-video extraction.
     */
    struct NativeVideoManifest
    {
        /** @brief Source video identity represented by this manifest. */
        std::string video_id;
        /** @brief Terminal native extraction status. */
        std::string status;
        /** @brief Measured decoded duration in milliseconds. */
        std::int64_t duration_ms;
        /** @brief Number of one-FPS targets expected from the video. */
        std::uint64_t expected_frame_count;
        /** @brief Number of durable frame rows actually emitted. */
        std::uint64_t emitted_frame_count;
        /** @brief Numeric average FPS from the decoded video stream. */
        double avg_fps;
        /** @brief Exact average FPS rational reported by FFmpeg. */
        RationalValue avg_fps_rational;
        /** @brief Native extractor version that produced the bundle. */
        std::string extractor_version;
        /** @brief Configuration hash used to produce the bundle. */
        std::string config_hash;
        /** @brief Relative path to the bundle's frame JSONL artifact. */
        std::string frames_jsonl;
    };

} // namespace hcmai::keyframes_extraction
