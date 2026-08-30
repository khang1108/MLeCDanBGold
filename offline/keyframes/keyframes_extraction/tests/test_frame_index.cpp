// Verify competition-facing coordinates and deterministic internal identities.

#include "hcmai/keyframes_extraction/frame_index.hpp"

#include "test_support.hpp"

#include <cstdint>
#include <limits>
#include <string>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Exercises the frame coordinate and internal identity contracts.
 *
 * @param None This test receives no command-line arguments.
 * @return Zero when all assertions pass.
 */
int run_frame_index_tests() {
    using test_support::require_throws;
    using test_support::require_true;

    require_true(submission_frame_idx(25.0, 0) == 0, "zero timestamp");
    require_true(submission_frame_idx(25.0, 1000) == 25, "25 FPS");
    require_true(
        submission_frame_idx(29.97, 1000) == 30,
        "29.97 FPS ceiling"
    );
    require_true(
        submission_frame_idx(30.0, 1500) == 45,
        "30 FPS fractional second"
    );
    require_true(
        make_frame_id("L01_V001", 0) == "L01_V001_raw1fps_000000000",
        "first internal ID"
    );
    require_true(
        make_frame_id("L01_V001", 12) == "L01_V001_raw1fps_000000012",
        "zero-padded internal ID"
    );

    require_throws([&] { submission_frame_idx(0.0, 1000); });
    require_throws([&] { submission_frame_idx(25.0, -1); });
    require_throws([&] {
        submission_frame_idx(std::numeric_limits<double>::quiet_NaN(), 1000);
    });
    require_throws([&] {
        submission_frame_idx(std::numeric_limits<double>::max(), 1);
    });
    require_throws([&] { make_frame_id("", 0); });
    require_throws([&] { make_frame_id("video/unsafe", 0); });

    return test_support::finish_test();
}

}  // namespace
}  // namespace hcmai::keyframes_extraction

/**
 * @brief Runs the frame index unit test executable.
 *
 * @param argc The number of command-line arguments; unused.
 * @param argv The command-line arguments; unused.
 * @return Zero when all assertions pass.
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    return hcmai::keyframes_extraction::run_frame_index_tests();
}
