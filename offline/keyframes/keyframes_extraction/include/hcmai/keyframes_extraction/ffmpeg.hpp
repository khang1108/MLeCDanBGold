/**
 * @file ffmpeg.hpp
 * @brief Declares FFmpeg-backed video decoding and JPEG encoding contracts.
 *
 * This header provides actual presentation timestamps and durable image
 * encoding. It does not perform sampling, source downloading, or publication.
 */

#pragma once

#include "hcmai/keyframes_extraction/types.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

struct AVFrame;

namespace hcmai::keyframes_extraction {

/**
 * @brief Configures one JPEG output representation.
 */
struct ImageVariant {
    /** @brief Maximum encoded long edge in pixels; zero preserves dimensions. */
    int max_long_edge;
    /** @brief Requested JPEG quality from 1 (lowest) through 100 (highest). */
    int quality;
};

/**
 * @brief Describes the image written by encode_jpeg.
 */
struct EncodedImage {
    /** @brief Final JPEG width in pixels. */
    int width;
    /** @brief Final JPEG height in pixels. */
    int height;
    /** @brief Final JPEG file size in bytes. */
    std::uint64_t bytes;
};

/**
 * @brief Retains authoritative FFmpeg stream metadata for one video.
 */
struct VideoInfo {
    /** @brief Numeric value of AVStream::avg_frame_rate. */
    double avg_fps;
    /** @brief Exact AVStream::avg_frame_rate numerator and denominator. */
    RationalValue avg_fps_rational;
    /** @brief FFmpeg stream time base used to interpret decoded PTS values. */
    RationalValue time_base;
    /** @brief Stream duration in milliseconds, or -1 when FFmpeg omits it. */
    std::int64_t duration_ms;
    /** @brief Decoded source width in pixels. */
    int width;
    /** @brief Decoded source height in pixels. */
    int height;
};

/**
 * @brief Holds one decoded video frame and its actual presentation metadata.
 */
struct DecodedFrame {
    /** @brief Zero-based order in which VideoDecoder returned this frame. */
    std::uint64_t ordinal;
    /** @brief Raw FFmpeg best-effort presentation timestamp. */
    std::int64_t pts;
    /** @brief Actual presentation timestamp converted to milliseconds. */
    std::int64_t timestamp_ms;
    /** @brief Shared RAII owner of the decoded AVFrame image data. */
    std::shared_ptr<AVFrame> image;
};

/**
 * @brief Decodes the first video stream of one local source video.
 *
 * The decoder owns its FFmpeg format/codec state and returns a separately
 * refcounted AVFrame so a caller may retain it across subsequent next() calls.
 */
class VideoDecoder {
public:
    /**
     * @brief Opens a local video, validates its first video stream, and prepares decoding.
     *
     * @param video_path Local source video path.
     * @return None; constructors do not return values.
     * @throws std::runtime_error If FFmpeg cannot open, inspect, or initialize video_path.
     */
    explicit VideoDecoder(const std::filesystem::path& video_path);

    /**
     * @brief Releases all owned FFmpeg format, codec, packet, and frame state.
     *
     * @return None; destructors do not return values.
     */
    ~VideoDecoder();

    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    /**
     * @brief Transfers decoder ownership to a new instance.
     *
     * @param other Decoder whose FFmpeg state is transferred.
     * @return None; constructors do not return values.
     */
    VideoDecoder(VideoDecoder&& other) noexcept;

    /**
     * @brief Replaces this decoder's FFmpeg state by moving another decoder.
     *
     * @param other Decoder whose FFmpeg state is transferred.
     * @return This decoder after ownership transfer.
     */
    VideoDecoder& operator=(VideoDecoder&& other) noexcept;

    /**
     * @brief Returns metadata captured from the selected FFmpeg video stream.
     *
     * @return Immutable authoritative metadata for the local source video.
     */
    const VideoInfo& info() const noexcept;

    /**
     * @brief Decodes the next presentation frame from the selected video stream.
     *
     * @return A RAII-owned decoded frame, or std::nullopt after decoder flush.
     * @throws std::runtime_error If FFmpeg decoding fails or a usable timestamp
     *                             cannot be produced for the source video.
     */
    std::optional<DecodedFrame> next();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Rescales a decoded frame and writes it as a JPEG image.
 *
 * @param source Decoded source AVFrame to convert; it must have positive dimensions.
 * @param output_path Destination JPEG path whose parent directory must exist.
 * @param variant JPEG dimension and quality settings.
 * @return Final encoded dimensions and file byte count.
 * @throws std::invalid_argument If source dimensions or variant values are invalid.
 * @throws std::runtime_error If scaling, JPEG encoding, or writing fails.
 */
EncodedImage encode_jpeg(
    const AVFrame& source,
    const std::filesystem::path& output_path,
    const ImageVariant& variant
);

}  // namespace hcmai::keyframes_extraction
