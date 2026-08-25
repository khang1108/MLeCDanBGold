/**
 * @file timestamp_sampler.cpp
 * @brief Implements deterministic nearest-frame selection for fixed targets.
 *
 * The sampler converts a monotonic decoded-frame timeline into one selection
 * per crossed target timestamp. It does not decode frames or calculate the
 * competition-facing frame_idx.
 */

#include "hcmai/keyframes_extraction/timestamp_sampler.hpp"

#include <limits>
#include <stdexcept>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Calculates the non-negative distance between two valid timestamps.
 *
 * @param left_ms The first non-negative timestamp in milliseconds.
 * @param right_ms The second non-negative timestamp in milliseconds.
 * @return The absolute difference as an unsigned 64-bit value.
 */
std::uint64_t timestamp_distance(
    std::int64_t left_ms,
    std::int64_t right_ms
) {
    if (left_ms >= right_ms) {
        return static_cast<std::uint64_t>(left_ms - right_ms);
    }

    return static_cast<std::uint64_t>(right_ms - left_ms);
}

}  // namespace

/**
 * @brief Creates a sampler that advances by one fixed target period.
 *
 * @param sample_period_ms The positive sampling interval in milliseconds.
 * @return None; constructors do not return values. The initialized sampler's
 *         first target is zero milliseconds.
 * @throws std::invalid_argument If sample_period_ms is not positive.
 */
TimestampSampler::TimestampSampler(std::int64_t sample_period_ms)
    : sample_period_ms_(sample_period_ms) {
    if (sample_period_ms_ <= 0) {
        throw std::invalid_argument("sample period must be positive");
    }
}

/**
 * @brief Emits one nearest decoded-frame selection for every crossed target.
 *
 * @param current The next decoded frame, with a non-negative timestamp that
 *                is not earlier than the preceding pushed frame.
 * @return Newly crossed targets in ascending target timestamp order. An exact
 *         previous/current tie selects the earlier frame.
 * @throws std::invalid_argument If current is negative or timestamps decrease.
 * @throws std::overflow_error If advancing to another target overflows int64.
 */
std::vector<SelectedTarget> TimestampSampler::push(const TimedFrame& current) {
    if (current.timestamp_ms < 0) {
        throw std::invalid_argument("decoded timestamp must be non-negative");
    }
    if (previous_.has_value() &&
        current.timestamp_ms < previous_->timestamp_ms) {
        throw std::invalid_argument("decoded timestamps must be monotonic");
    }

    std::vector<SelectedTarget> selected;
    const std::int64_t maximum_timestamp =
        std::numeric_limits<std::int64_t>::max();

    while (current.timestamp_ms >= next_target_ms_) {
        if (next_target_ms_ > maximum_timestamp - sample_period_ms_) {
            throw std::overflow_error("next target timestamp exceeds int64");
        }

        const TimedFrame* winner = &current;
        if (previous_.has_value()) {
            const std::uint64_t previous_distance = timestamp_distance(
                next_target_ms_,
                previous_->timestamp_ms
            );
            const std::uint64_t current_distance = timestamp_distance(
                current.timestamp_ms,
                next_target_ms_
            );
            if (previous_distance <= current_distance) {
                winner = &previous_.value();
            }
        }

        selected.push_back(SelectedTarget{
            static_cast<std::uint64_t>(
                next_target_ms_ / sample_period_ms_
            ),
            next_target_ms_,
            winner->ordinal,
            winner->timestamp_ms,
        });
        next_target_ms_ += sample_period_ms_;
    }

    previous_ = current;
    return selected;
}

}  // namespace hcmai::keyframes_extraction
