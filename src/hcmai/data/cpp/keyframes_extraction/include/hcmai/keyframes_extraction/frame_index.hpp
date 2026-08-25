#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace hcmai::keyframes_extraction
{

    /**
     * @brief Computes the competition-facing frame coordinate for a timestamp.
     *
     * The coordinate follows the competition contract:
     * floor(ceil(avg_fps) * timestamp_ms / 1000).
     *
     * @param avg_fps The video stream average frame rate; it must be finite and
     *                positive.
     * @param timestamp_ms The selected frame timestamp in milliseconds; it must
     *                     be non-negative.
     * @return The non-negative competition frame index.
     * @throws std::invalid_argument If the FPS or timestamp is invalid.
     * @throws std::overflow_error If the result does not fit in int64_t.
     */
    std::int64_t submission_frame_idx(
        double avg_fps,
        std::int64_t timestamp_ms);

    /**
     * @brief Builds the deterministic internal identity for a sampled frame.
     *
     * This identity is internal to the custom extraction pipeline and is
     * intentionally independent of the competition-facing frame_idx value.
     *
     * @param video_id The source video identifier; it may contain only letters,
     *                 digits, underscore, hyphen, and period.
     * @param sample_index The zero-based one-FPS sample ordinal.
     * @return An ID formatted as {video_id}_raw1fps_{sample_index:09d}.
     * @throws std::invalid_argument If video_id is blank or contains unsafe
     *                               characters.
     */
    std::string make_frame_id(
        std::string_view video_id,
        std::uint64_t sample_index);

} // namespace hcmai::keyframes_extraction
