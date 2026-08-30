// Verify nearest-frame selection for deterministic one-FPS targets.

#include "hcmai/keyframes_extraction/timestamp_sampler.hpp"

#include "test_support.hpp"

#include <cstdint>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Exercises target crossing, nearest-frame selection, and validation.
 *
 * @param None This test receives no command-line arguments.
 * @return Zero when all assertions pass.
 */
int run_timestamp_sampler_tests() {
    using test_support::require_throws;
    using test_support::require_true;

    TimestampSampler sampler(1000);
    const auto first = sampler.push(TimedFrame{0, 0});
    require_true(first.size() == 1, "target zero must emit");
    require_true(
        first[0].selected_ordinal == 0,
        "first frame selected for target zero"
    );

    const auto second = sampler.push(TimedFrame{1, 400});
    require_true(second.empty(), "target one second is not crossed");

    const auto third = sampler.push(TimedFrame{2, 1600});
    require_true(third.size() == 1, "target one second must emit");
    require_true(
        third[0].selected_ordinal == 1,
        "nearest previous frame selected"
    );
    require_true(
        third[0].selected_timestamp_ms == 400,
        "selected timestamp is retained"
    );

    TimestampSampler tie(1000);
    tie.push(TimedFrame{0, 500});
    const auto tie_result = tie.push(TimedFrame{1, 1500});
    require_true(
        tie_result[0].selected_ordinal == 0,
        "exact tie chooses earlier frame"
    );

    TimestampSampler gap(1000);
    gap.push(TimedFrame{0, 0});
    const auto crossed = gap.push(TimedFrame{1, 3500});
    require_true(crossed.size() == 3, "all crossed targets emitted exactly once");
    require_true(
        crossed[0].target_timestamp_ms == 1000,
        "first crossed target"
    );
    require_true(
        crossed[2].target_timestamp_ms == 3000,
        "last crossed target"
    );
    require_true(crossed[0].sample_index == 1, "first crossed sample index");
    require_true(crossed[1].sample_index == 2, "second crossed sample index");
    require_true(crossed[2].sample_index == 3, "third crossed sample index");

    require_throws([&] { TimestampSampler invalid(0); });
    require_throws([&] { sampler.push(TimedFrame{3, -1}); });
    require_throws([&] { sampler.push(TimedFrame{4, 300}); });

    return test_support::finish_test();
}

}  // namespace
}  // namespace hcmai::keyframes_extraction

/**
 * @brief Runs the timestamp sampler unit test executable.
 *
 * @param argc The number of command-line arguments; unused.
 * @param argv The command-line arguments; unused.
 * @return Zero when all assertions pass.
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    return hcmai::keyframes_extraction::run_timestamp_sampler_tests();
}
