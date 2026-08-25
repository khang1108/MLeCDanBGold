/**
 * @file frame_index.cpp
 * @brief Implements competition frame coordinates and custom internal IDs.
 *
 * This module owns only deterministic identity calculations. It does not
 * decode video frames, infer timestamps, or inspect organizer keyframes.
 */

#include "hcmai/keyframes_extraction/frame_index.hpp"

#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Checks whether a video ID is safe to embed in a filesystem-facing ID.
 *
 * @param video_id The candidate video identifier.
 * @return True when the ID is non-blank and contains only letters, digits,
 *         underscore, hyphen, or period; otherwise false.
 */
bool is_safe_video_id(std::string_view video_id) {
    if (video_id.empty()) {
        return false;
    }

    for (const unsigned char character : video_id) {
        if (std::isalnum(character) == 0 && character != '_' &&
            character != '-' && character != '.') {
            return false;
        }
    }

    return true;
}

}  // namespace

/**
 * @brief Computes the organizer-required frame coordinate from actual time.
 *
 * @param avg_fps The finite, positive AVStream average frame rate.
 * @param timestamp_ms The selected actual presentation timestamp in
 *                     milliseconds.
 * @return floor(ceil(avg_fps) * timestamp_ms / 1000) as a non-negative
 *         signed 64-bit frame index.
 * @throws std::invalid_argument If avg_fps is non-finite/non-positive or the
 *                               timestamp is negative.
 * @throws std::overflow_error If the calculated coordinate cannot fit in an
 *                             int64_t.
 */
std::int64_t submission_frame_idx(double avg_fps, std::int64_t timestamp_ms) {
    if (!std::isfinite(avg_fps) || avg_fps <= 0.0 || timestamp_ms < 0) {
        throw std::invalid_argument("invalid FPS or timestamp");
    }

    const long double fps_ceiling = std::ceil(static_cast<long double>(avg_fps));
    const long double value = std::floor(
        fps_ceiling * static_cast<long double>(timestamp_ms) / 1000.0L
    );
    const long double maximum = static_cast<long double>(
        std::numeric_limits<std::int64_t>::max()
    );

    if (!std::isfinite(value) || value < 0.0L || value > maximum) {
        throw std::overflow_error("frame_idx exceeds int64");
    }

    return static_cast<std::int64_t>(value);
}

/**
 * @brief Builds the stable internal ID for a one-FPS sample.
 *
 * @param video_id The source video identifier, restricted to safe filename
 *                 characters.
 * @param sample_index The zero-based ordinal of the one-FPS target.
 * @return The exact internal ID {video_id}_raw1fps_{sample_index:09d}.
 * @throws std::invalid_argument If video_id is blank or unsafe.
 */
std::string make_frame_id(
    std::string_view video_id,
    std::uint64_t sample_index
) {
    if (!is_safe_video_id(video_id)) {
        throw std::invalid_argument("video_id must be non-blank and safe");
    }

    std::ostringstream output;
    output << video_id << "_raw1fps_" << std::setfill('0') << std::setw(9)
           << sample_index;
    return output.str();
}

}  // namespace hcmai::keyframes_extraction
