/**
 * @file timestamp_sampler.hpp
 * @brief Declares deterministic nearest-frame selection at fixed timestamps.
 *
 * This contract operates on decoded timestamps only. It does not decode video
 * or assign competition-facing frame coordinates.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace hcmai::keyframes_extraction
{

    /**
     * @brief Represents one decoded video frame on the presentation timeline.
     *
     * @param ordinal The monotonic decode ordinal used for internal selection.
     * @param timestamp_ms The non-negative presentation timestamp in milliseconds.
     */
    struct TimedFrame
    {
        std::uint64_t ordinal;
        std::int64_t timestamp_ms;
    };

    /**
     * @brief Records which decoded frame represents one sampling target.
     *
     * @param sample_index The zero-based target index at the configured period.
     * @param target_timestamp_ms The requested target timestamp.
     * @param selected_ordinal The ordinal of the selected decoded frame.
     * @param selected_timestamp_ms The actual timestamp of the selected frame.
     */
    struct SelectedTarget
    {
        std::uint64_t sample_index;
        std::int64_t target_timestamp_ms;
        std::uint64_t selected_ordinal;
        std::int64_t selected_timestamp_ms;
    };

    /**
     * @brief Selects the nearest decoded frame for each crossed target timestamp.
     *
     * The sampler emits target zero, preserves monotonic target order, emits each
     * target once, and resolves an exact previous/current distance tie in favor of
     * the earlier decoded frame.
     */
    class TimestampSampler
    {
    public:
        /**
         * @brief Creates a sampler for a fixed target period.
         *
         * @param sample_period_ms The positive interval between target timestamps.
         * @throws std::invalid_argument If the period is not positive.
         */
        explicit TimestampSampler(std::int64_t sample_period_ms);

        /**
         * @brief Consumes one decoded frame and returns newly crossed targets.
         *
         * @param current The next decoded frame; its timestamp must be
         *                non-negative and monotonic.
         * @return All target selections crossed by current, in target order.
         * @throws std::invalid_argument If current has a negative or decreasing
         *                               timestamp.
         * @throws std::overflow_error If the next target cannot be represented.
         */
        std::vector<SelectedTarget> push(const TimedFrame &current);

    private:
        std::int64_t sample_period_ms_;
        std::int64_t next_target_ms_ = 0;
        std::optional<TimedFrame> previous_;
    };

} // namespace hcmai::keyframes_extraction
