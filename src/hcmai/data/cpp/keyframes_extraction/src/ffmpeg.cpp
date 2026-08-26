/**
 * @file ffmpeg.cpp
 * @brief Implements FFmpeg video decoding, timestamp conversion, and JPEG output.
 *
 * This module owns FFmpeg resource lifetime and image conversion. It does not
 * select one-FPS targets, acquire source videos, or publish native artifacts.
 */

#include "hcmai/keyframes_extraction/ffmpeg.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Formats one FFmpeg error code with source-path context.
 *
 * @param operation Short description of the failed FFmpeg operation.
 * @param path Local input or output path associated with the operation.
 * @param error_code Negative FFmpeg error code.
 * @return Human-readable exception message containing operation and path.
 */
std::string ffmpeg_error_message(
    const std::string& operation,
    const std::filesystem::path& path,
    int error_code
) {
    char error_buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(error_code, error_buffer, sizeof(error_buffer));
    return operation + " for '" + path.string() + "': " + error_buffer;
}

/**
 * @brief Throws a runtime error for one failed FFmpeg operation.
 *
 * @param operation Short description of the failed FFmpeg operation.
 * @param path Local input or output path associated with the operation.
 * @param error_code Negative FFmpeg error code.
 * @return None; this function always throws std::runtime_error.
 * @throws std::runtime_error Always, with FFmpeg and path context.
 */
[[noreturn]] void throw_ffmpeg_error(
    const std::string& operation,
    const std::filesystem::path& path,
    int error_code
) {
    throw std::runtime_error(ffmpeg_error_message(operation, path, error_code));
}

/**
 * @brief Frees an FFmpeg frame through a pointer-compatible deleter.
 *
 * @param frame Frame to free; null is accepted.
 * @return None; releases frame resources when present.
 */
void free_frame(AVFrame* frame) noexcept {
    if (frame != nullptr) {
        av_frame_free(&frame);
    }
}

/**
 * @brief Frees an FFmpeg packet through a pointer-compatible deleter.
 *
 * @param packet Packet to free; null is accepted.
 * @return None; releases packet resources when present.
 */
void free_packet(AVPacket* packet) noexcept {
    if (packet != nullptr) {
        av_packet_free(&packet);
    }
}

/**
 * @brief Frees an FFmpeg codec context through a pointer-compatible deleter.
 *
 * @param context Codec context to free; null is accepted.
 * @return None; releases codec resources when present.
 */
void free_codec_context(AVCodecContext* context) noexcept {
    if (context != nullptr) {
        avcodec_free_context(&context);
    }
}

/**
 * @brief Frees a libswscale context through a pointer-compatible deleter.
 *
 * @param context Scaling context to free; null is accepted.
 * @return None; releases scaling resources when present.
 */
void free_scaler_context(SwsContext* context) noexcept {
    if (context != nullptr) {
        sws_freeContext(context);
    }
}

using FrameHandle = std::unique_ptr<AVFrame, decltype(&free_frame)>;
using PacketHandle = std::unique_ptr<AVPacket, decltype(&free_packet)>;
using CodecContextHandle = std::unique_ptr<
    AVCodecContext,
    decltype(&free_codec_context)
>;
using ScalerHandle = std::unique_ptr<
    SwsContext,
    decltype(&free_scaler_context)
>;

/**
 * @brief Converts a valid FFmpeg timestamp to nearest milliseconds.
 *
 * @param pts Raw best-effort presentation timestamp.
 * @param time_base FFmpeg stream time base for pts.
 * @param video_path Local source video path used in errors.
 * @return Non-negative timestamp in milliseconds.
 * @throws std::runtime_error If pts is unavailable, invalid, or negative.
 */
std::int64_t timestamp_to_milliseconds(
    std::int64_t pts,
    AVRational time_base,
    const std::filesystem::path& video_path
) {
    if (pts == AV_NOPTS_VALUE || time_base.num <= 0 || time_base.den <= 0) {
        throw std::runtime_error(
            "decoded frame has no usable timestamp for '" + video_path.string() + "'"
        );
    }

    const AVRational milliseconds = {1, 1000};
    const AVRounding rounding = static_cast<AVRounding>(
        AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX
    );
    const std::int64_t timestamp_ms = av_rescale_q_rnd(
        pts,
        time_base,
        milliseconds,
        rounding
    );
    if (timestamp_ms == AV_NOPTS_VALUE || timestamp_ms < 0) {
        throw std::runtime_error(
            "decoded frame timestamp is invalid for '" + video_path.string() + "'"
        );
    }

    return timestamp_ms;
}

/**
 * @brief Converts a stream duration to milliseconds when FFmpeg provides one.
 *
 * @param duration Stream duration in its native time base.
 * @param time_base FFmpeg stream time base for duration.
 * @return Non-negative milliseconds, or -1 when duration is unavailable.
 */
std::int64_t duration_to_milliseconds(
    std::int64_t duration,
    AVRational time_base
) {
    if (duration == AV_NOPTS_VALUE || duration < 0 ||
        time_base.num <= 0 || time_base.den <= 0) {
        return -1;
    }

    const AVRational milliseconds = {1, 1000};
    const AVRounding rounding = static_cast<AVRounding>(
        AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX
    );
    const std::int64_t duration_ms = av_rescale_q_rnd(
        duration,
        time_base,
        milliseconds,
        rounding
    );
    return duration_ms < 0 || duration_ms == AV_NOPTS_VALUE ? -1 : duration_ms;
}

/**
 * @brief Validates JPEG quality and output-edge configuration.
 *
 * @param variant Requested JPEG dimensions and quality.
 * @param output_path Destination path used in exceptions.
 * @return None; throws when the image variant is invalid.
 * @throws std::invalid_argument If max_long_edge or quality is unsupported.
 */
void validate_image_variant(
    const ImageVariant& variant,
    const std::filesystem::path& output_path
) {
    if (variant.max_long_edge < 0) {
        throw std::invalid_argument(
            "JPEG max_long_edge must be non-negative for '" +
            output_path.string() + "'"
        );
    }
    if (variant.quality < 1 || variant.quality > 100) {
        throw std::invalid_argument(
            "JPEG quality must be between 1 and 100 for '" +
            output_path.string() + "'"
        );
    }
}

/**
 * @brief Verifies the destination's parent already exists as a directory.
 *
 * @param output_path Destination JPEG path.
 * @return None; allows writing when output_path has a valid parent.
 * @throws std::invalid_argument If output_path is blank.
 * @throws std::runtime_error If the parent does not exist or is not a directory.
 */
void validate_output_parent(const std::filesystem::path& output_path) {
    if (output_path.empty()) {
        throw std::invalid_argument("JPEG output path must not be blank");
    }

    const std::filesystem::path parent = output_path.parent_path();
    if (parent.empty()) {
        return;
    }

    std::error_code error;
    if (!std::filesystem::is_directory(parent, error)) {
        throw std::runtime_error(
            "JPEG output parent is unavailable for '" + output_path.string() + "'"
        );
    }
}

/**
 * @brief Calculates output dimensions while preserving the source aspect ratio.
 *
 * @param source_width Positive source image width.
 * @param source_height Positive source image height.
 * @param max_long_edge Non-negative output long-edge limit.
 * @param output_path Destination path used in exceptions.
 * @return Positive output width and height, bounded when a limit is active.
 * @throws std::invalid_argument If source dimensions are invalid.
 * @throws std::overflow_error If scaled dimensions cannot fit in int.
 */
std::pair<int, int> scaled_dimensions(
    int source_width,
    int source_height,
    int max_long_edge,
    const std::filesystem::path& output_path
) {
    if (source_width <= 0 || source_height <= 0) {
        throw std::invalid_argument(
            "JPEG source dimensions must be positive for '" +
            output_path.string() + "'"
        );
    }

    const int source_long_edge = std::max(source_width, source_height);
    if (max_long_edge == 0 || source_long_edge <= max_long_edge) {
        return {source_width, source_height};
    }

    const long double scale = static_cast<long double>(max_long_edge) /
        static_cast<long double>(source_long_edge);
    const long long scaled_width = std::llround(
        static_cast<long double>(source_width) * scale
    );
    const long long scaled_height = std::llround(
        static_cast<long double>(source_height) * scale
    );

    if (scaled_width <= 0 || scaled_height <= 0 ||
        scaled_width > std::numeric_limits<int>::max() ||
        scaled_height > std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "JPEG scaled dimensions are invalid for '" + output_path.string() + "'"
        );
    }

    return {
        static_cast<int>(scaled_width),
        static_cast<int>(scaled_height),
    };
}

/**
 * @brief Maps a human JPEG quality value to FFmpeg's MJPEG quantizer range.
 *
 * @param quality Quality from 1 (lowest) through 100 (highest).
 * @return MJPEG quantizer from 31 (lowest) through 2 (highest).
 */
int jpeg_quantizer(int quality) {
    return 2 + ((100 - quality) * 29 + 49) / 99;
}

/**
 * @brief Wraps an AVFrame clone in shared RAII ownership.
 *
 * @param frame Owned AVFrame pointer returned by av_frame_clone.
 * @return Shared owner that frees frame through av_frame_free.
 */
std::shared_ptr<AVFrame> shared_frame(AVFrame* frame) {
    return std::shared_ptr<AVFrame>(frame, free_frame);
}

}  // namespace

/**
 * @brief Holds all FFmpeg state required by one active VideoDecoder.
 */
class VideoDecoder::Impl {
public:
    /**
     * @brief Initializes empty FFmpeg state for one local source path.
     *
     * @param source_path Local video path used for decoder errors.
     * @return None; constructors do not return values.
     */
    explicit Impl(std::filesystem::path source_path)
        : video_path(std::move(source_path)) {}

    /**
     * @brief Releases FFmpeg resources in reverse dependency order.
     *
     * @return None; destructors do not return values.
     */
    ~Impl() {
        av_frame_free(&decode_frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        if (format_context != nullptr) {
            avformat_close_input(&format_context);
        }
    }

    /** @brief Local source path used in all decoder error messages. */
    std::filesystem::path video_path;
    /** @brief Opened FFmpeg demuxer context. */
    AVFormatContext* format_context = nullptr;
    /** @brief Opened FFmpeg decoder context for the selected stream. */
    AVCodecContext* codec_context = nullptr;
    /** @brief Reusable packet storage for demuxed packets. */
    AVPacket* packet = nullptr;
    /** @brief Reusable decoder output frame storage. */
    AVFrame* decode_frame = nullptr;
    /** @brief Index of the first source video stream. */
    int video_stream_index = -1;
    /** @brief FFmpeg time base used to rescale best-effort timestamps. */
    AVRational time_base = {0, 1};
    /** @brief Stream metadata exposed through VideoDecoder::info. */
    VideoInfo video_info = {};
    /** @brief Whether a null packet has been sent to flush the decoder. */
    bool flushing = false;
    /** @brief Whether all frames have been received after decoder flush. */
    bool exhausted = false;
    /** @brief Ordinal assigned to the next decoded frame returned to the caller. */
    std::uint64_t next_ordinal = 0;
};

/**
 * @brief Opens and initializes FFmpeg decoding for the first video stream.
 *
 * @param video_path Local source video path.
 * @return None; constructors do not return values.
 * @throws std::invalid_argument If video_path is blank.
 * @throws std::runtime_error If FFmpeg cannot inspect or initialize the video.
 */
VideoDecoder::VideoDecoder(const std::filesystem::path& video_path) {
    if (video_path.empty()) {
        throw std::invalid_argument("video path must not be blank");
    }

    auto implementation = std::make_unique<Impl>(video_path);
    const std::string path_text = video_path.string();

    // Open the source video and read its stream information.
    // Return a value < 0 on failure, and a positive value on success.
    int result = avformat_open_input(
        &implementation->format_context,
        path_text.c_str(),
        nullptr,
        nullptr
    );
    if (result < 0) {
        throw_ffmpeg_error("unable to open video", video_path, result);
    }

    // Retrieve stream information.
    result = avformat_find_stream_info(implementation->format_context, nullptr);
    if (result < 0) {
        throw_ffmpeg_error("unable to inspect video streams", video_path, result);
    }

    // Loop through all streams to find the first video stream.
    AVStream* video_stream = nullptr;
    for (unsigned int index = 0;
            index < implementation->format_context->nb_streams;
            ++index) {

        // Get the candidate stream and verify it is a video stream with codec parameters.
        AVStream* candidate = implementation->format_context->streams[index];

        // If the candidate is a video stream, record it and stop searching.
        if (candidate != nullptr && candidate->codecpar != nullptr &&
            candidate->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = candidate;
            implementation->video_stream_index = static_cast<int>(index);
            break;
        }
    }
    if (video_stream == nullptr) {
        throw std::runtime_error("no video stream found in '" + video_path.string() + "'");
    }

    const AVRational avg_frame_rate = video_stream->avg_frame_rate;
    const double avg_fps = av_q2d(avg_frame_rate);
    if (avg_frame_rate.num <= 0 || avg_frame_rate.den <= 0 ||
        !std::isfinite(avg_fps) || avg_fps <= 0.0) {
        throw std::runtime_error(
            "video stream has invalid average FPS in '" + video_path.string() + "'"
        );
    }
    if (video_stream->time_base.num <= 0 || video_stream->time_base.den <= 0) {
        throw std::runtime_error(
            "video stream has invalid time base in '" + video_path.string() + "'"
        );
    }

    // Find the decoder for the video stream and allocate a codec context.
    // A decoder is required to convert compressed packets into raw frames.
    const AVCodec* decoder = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (decoder == nullptr) {
        throw std::runtime_error(
            "no decoder available for video stream in '" + video_path.string() + "'"
        );
    }

    implementation->codec_context = avcodec_alloc_context3(decoder);
    if (implementation->codec_context == nullptr) {
        throw std::runtime_error(
            "unable to allocate decoder context for '" + video_path.string() + "'"
        );
    }
    // Copy the codec parameters from the video stream to the
    // codec context and open the decoder.
    result = avcodec_parameters_to_context(
        implementation->codec_context,
        video_stream->codecpar
    );
    if (result < 0) {
        throw_ffmpeg_error("unable to copy video codec parameters", video_path, result);
    }

    // Open the decoder to prepare for decoding packets into frames.
    result = avcodec_open2(implementation->codec_context, decoder, nullptr);
    if (result < 0) {
        throw_ffmpeg_error("unable to open video decoder", video_path, result);
    }

    if (implementation->codec_context->width <= 0 ||
        implementation->codec_context->height <= 0) {
        throw std::runtime_error(
            "video stream has invalid dimensions in '" + video_path.string() + "'"
        );
    }

    implementation->packet = av_packet_alloc();
    implementation->decode_frame = av_frame_alloc();

    if (implementation->packet == nullptr || implementation->decode_frame == nullptr) {
        throw std::runtime_error(
            "unable to allocate decoder packet or frame for '" + video_path.string() + "'"
        );
    }

    // Record the stream time base and metadata for later use
    // in decoding and timestamp conversion.
    implementation->time_base = video_stream->time_base;
    implementation->video_info = VideoInfo{
        avg_fps,
        RationalValue{
            static_cast<std::int64_t>(avg_frame_rate.num),
            static_cast<std::int64_t>(avg_frame_rate.den),
        },
        RationalValue{
            static_cast<std::int64_t>(video_stream->time_base.num),
            static_cast<std::int64_t>(video_stream->time_base.den),
        },
        duration_to_milliseconds(video_stream->duration, video_stream->time_base),
        implementation->codec_context->width,
        implementation->codec_context->height,
    };
    impl_ = std::move(implementation);
}

/**
 * @brief Releases all FFmpeg state owned by this decoder.
 *
 * @return None; destructors do not return values.
 */
VideoDecoder::~VideoDecoder() = default;

/**
 * @brief Transfers all FFmpeg state from another decoder.
 *
 * @param other Decoder whose state is transferred.
 * @return None; constructors do not return values.
 */
VideoDecoder::VideoDecoder(VideoDecoder&& other) noexcept = default;

/**
 * @brief Replaces this decoder state by moving another decoder.
 *
 * @param other Decoder whose state is transferred.
 * @return This decoder after state transfer.
 */
VideoDecoder& VideoDecoder::operator=(VideoDecoder&& other) noexcept = default;

/**
 * @brief Returns metadata recorded while opening the source video.
 *
 * @return Immutable FFmpeg-derived metadata for the selected video stream.
 */
const VideoInfo& VideoDecoder::info() const noexcept {
    return impl_->video_info;
}

/**
 * @brief Decodes and returns the next frame from the selected video stream.
 *
 * @return A refcounted decoded frame, or std::nullopt after a full decoder flush.
 * @throws std::logic_error If called on a moved-from decoder.
 * @throws std::runtime_error If demuxing, decoding, cloning, or timestamp
 *                             conversion fails.
 */
std::optional<DecodedFrame> VideoDecoder::next() {
    if (!impl_) {
        throw std::logic_error("VideoDecoder has no active source");
    }
    if (impl_->exhausted) {
        return std::nullopt;
    }

    /**
     * Decodes frames in a loop until a frame is successfully received,
     * the decoder is exhausted, or an error occurs.
     */
    while (true) {
        const int receive_result = avcodec_receive_frame(
            impl_->codec_context,
            impl_->decode_frame
        );

        if (receive_result == 0) {
            const std::int64_t pts = impl_->decode_frame->best_effort_timestamp;
            try {
                const std::int64_t timestamp_ms = timestamp_to_milliseconds(
                    pts,
                    impl_->time_base,
                    impl_->video_path
                );

                // Clone the decoded frame to a shared pointer for RAII ownership.
                AVFrame* cloned = av_frame_clone(impl_->decode_frame);
                if (cloned == nullptr) {
                    throw std::runtime_error(
                        "unable to clone decoded frame for '" +
                        impl_->video_path.string() + "'"
                    );
                }

                // Share the cloned frame so it is freed when no longer used.
                const std::shared_ptr<AVFrame> image = shared_frame(cloned);
                av_frame_unref(impl_->decode_frame); // Release the reusable decode frame for the next call.

                return DecodedFrame{
                    impl_->next_ordinal++,
                    pts,
                    timestamp_ms,
                    image,
                };
            } catch (...) {
                av_frame_unref(impl_->decode_frame);
                throw;
            }
        }
        if (receive_result == AVERROR_EOF) {
            impl_->exhausted = true;
            return std::nullopt;
        }
        if (receive_result != AVERROR(EAGAIN)) {
            throw_ffmpeg_error(
                "unable to receive decoded video frame",
                impl_->video_path,
                receive_result
            );
        }

        if (impl_->flushing) {
            throw std::runtime_error(
                "decoder stalled while flushing '" + impl_->video_path.string() + "'"
            );
        }

        const int read_result = av_read_frame(
            impl_->format_context,
            impl_->packet
        );
        if (read_result == AVERROR_EOF) {
            const int flush_result = avcodec_send_packet(
                impl_->codec_context,
                nullptr
            );
            if (flush_result < 0) {
                throw_ffmpeg_error(
                    "unable to flush video decoder",
                    impl_->video_path,
                    flush_result
                );
            }
            impl_->flushing = true;
            continue;
        }
        if (read_result < 0) {
            throw_ffmpeg_error(
                "unable to read video packet",
                impl_->video_path,
                read_result
            );
        }

        if (impl_->packet->stream_index != impl_->video_stream_index) {
            av_packet_unref(impl_->packet);
            continue;
        }

        const int send_result = avcodec_send_packet(
            impl_->codec_context,
            impl_->packet
        );
        av_packet_unref(impl_->packet);
        if (send_result < 0) {
            throw_ffmpeg_error(
                "unable to send video packet to decoder",
                impl_->video_path,
                send_result
            );
        }
    }
}

/**
 * @brief Scales one decoded frame and writes a JPEG file at the requested path.
 *
 * @param source Decoded FFmpeg frame to convert.
 * @param output_path Existing-parent destination JPEG path.
 * @param variant Maximum long-edge and quality settings.
 * @return Final JPEG dimensions and file byte count.
 * @throws std::invalid_argument If source, path, or variant values are invalid.
 * @throws std::runtime_error If FFmpeg scaling, encoding, or file output fails.
 */
EncodedImage encode_jpeg(
    const AVFrame& source,
    const std::filesystem::path& output_path,
    const ImageVariant& variant
) {
    validate_image_variant(variant, output_path);
    validate_output_parent(output_path);

    if (source.format == AV_PIX_FMT_NONE) {
        throw std::invalid_argument(
            "JPEG source pixel format is unavailable for '" +
            output_path.string() + "'"
        );
    }
    const auto [output_width, output_height] = scaled_dimensions(
        source.width,
        source.height,
        variant.max_long_edge,
        output_path
    );

    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (encoder == nullptr) {
        throw std::runtime_error(
            "MJPEG encoder is unavailable for '" + output_path.string() + "'"
        );
    }

    CodecContextHandle encoder_context(
        avcodec_alloc_context3(encoder),
        &free_codec_context
    );
    if (!encoder_context) {
        throw std::runtime_error(
            "unable to allocate JPEG encoder for '" + output_path.string() + "'"
        );
    }

    const int quantizer = jpeg_quantizer(variant.quality);

    encoder_context->pix_fmt = AV_PIX_FMT_YUVJ420P;

    encoder_context->width = output_width;
    encoder_context->height = output_height;

    encoder_context->time_base = AVRational{1, 1};

    encoder_context->flags |= AV_CODEC_FLAG_QSCALE;

    encoder_context->global_quality = quantizer * FF_QP2LAMBDA;

    encoder_context->qmin = quantizer;
    encoder_context->qmax = quantizer;

    int result = avcodec_open2(encoder_context.get(), encoder, nullptr);
    if (result < 0) {
        throw_ffmpeg_error("unable to open JPEG encoder", output_path, result);
    }

    FrameHandle encoded_frame(av_frame_alloc(), &free_frame);
    if (!encoded_frame) {
        throw std::runtime_error(
            "unable to allocate JPEG frame for '" + output_path.string() + "'"
        );
    }
    encoded_frame->format = encoder_context->pix_fmt;

    encoded_frame->width = output_width;
    encoded_frame->height = output_height;

    result = av_frame_get_buffer(encoded_frame.get(), 32);
    if (result < 0) {
        throw_ffmpeg_error("unable to allocate JPEG frame buffer", output_path, result);
    }
    result = av_frame_make_writable(encoded_frame.get());
    if (result < 0) {
        throw_ffmpeg_error("unable to make JPEG frame writable", output_path, result);
    }

    ScalerHandle scaler(
        sws_getContext(
            source.width,
            source.height,
            static_cast<AVPixelFormat>(source.format),
            output_width,
            output_height,
            encoder_context->pix_fmt,
            SWS_BICUBIC,
            nullptr,
            nullptr,
            nullptr
        ),
        &free_scaler_context
    );
    if (!scaler) {
        throw std::runtime_error(
            "unable to create JPEG scaler for '" + output_path.string() + "'"
        );
    }

    const int scaled_lines = sws_scale(
        scaler.get(),
        source.data,
        source.linesize,
        0,
        source.height,
        encoded_frame->data,
        encoded_frame->linesize
    );
    if (scaled_lines != output_height) {
        throw std::runtime_error(
            "unable to scale complete JPEG frame for '" + output_path.string() + "'"
        );
    }

    encoded_frame->pts = 0;
    result = avcodec_send_frame(encoder_context.get(), encoded_frame.get());
    if (result < 0) {
        throw_ffmpeg_error("unable to send JPEG frame to encoder", output_path, result);
    }

    PacketHandle encoded_packet(av_packet_alloc(), &free_packet);
    if (!encoded_packet) {
        throw std::runtime_error(
            "unable to allocate JPEG packet for '" + output_path.string() + "'"
        );
    }
    result = avcodec_receive_packet(encoder_context.get(), encoded_packet.get());
    if (result < 0) {
        throw_ffmpeg_error("unable to receive JPEG packet", output_path, result);
    }
    if (encoded_packet->data == nullptr || encoded_packet->size <= 0) {
        throw std::runtime_error(
            "JPEG encoder produced no bytes for '" + output_path.string() + "'"
        );
    }

    {
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "unable to open JPEG output '" + output_path.string() + "'"
            );
        }
        output.write(
            reinterpret_cast<const char*>(encoded_packet->data),
            encoded_packet->size
        );
        output.close();
        if (!output) {
            throw std::runtime_error(
                "unable to write JPEG output '" + output_path.string() + "'"
            );
        }
    }

    const std::uintmax_t file_bytes = std::filesystem::file_size(output_path);
    if (file_bytes > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "JPEG output is too large for byte accounting: '" +
            output_path.string() + "'"
        );
    }

    return EncodedImage{
        output_width,
        output_height,
        static_cast<std::uint64_t>(file_bytes),
    };
}

}  // namespace hcmai::keyframes_extraction
