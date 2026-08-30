/**
 * @file test_ffmpeg.cpp
 * @brief Verifies local FFmpeg decoding, timestamps, and JPEG encoding.
 *
 * The test creates a synthetic local source with an argv-only helper. This
 * deliberately stays test-local until Task 5 provides the production process
 * runner, so Task 4 remains scoped to the FFmpeg layer.
 */

#include "hcmai/keyframes_extraction/ffmpeg.hpp"

#include "test_support.hpp"

#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Runs a deterministic test fixture command without shell interpolation.
 *
 * @param arguments Program name followed by literal argv values.
 * @return The child's zero exit status.
 * @throws std::invalid_argument If arguments is empty.
 * @throws std::runtime_error If fork, waitpid, exec, or the child command fails.
 */
int run_fixture_argv(const std::vector<std::string>& arguments) {
    if (arguments.empty()) {
        throw std::invalid_argument("fixture command requires argv");
    }

    std::vector<char*> child_argv;
    child_argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
        child_argv.push_back(const_cast<char*>(argument.c_str()));
    }
    child_argv.push_back(nullptr);

    const pid_t child = fork();
    if (child < 0) {
        throw std::runtime_error("fork failed for FFmpeg fixture");
    }
    if (child == 0) {
        execvp(child_argv.front(), child_argv.data());
        _exit(127);
    }

    int status = 0;
    pid_t waited = 0;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);

    if (waited < 0) {
        throw std::runtime_error("waitpid failed for FFmpeg fixture");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("FFmpeg fixture command failed");
    }

    return WEXITSTATUS(status);
}

/**
 * @brief Creates the small local video used by FFmpeg integration assertions.
 *
 * @param directory Existing temporary directory for fixture output.
 * @return Filesystem path to a readable 80x40, 2-FPS MPEG-4 video.
 * @throws std::runtime_error If the local FFmpeg command fails.
 */
std::filesystem::path make_synthetic_video(
    const std::filesystem::path& directory
) {
    const std::filesystem::path video_path = directory / "synthetic.mp4";
    run_fixture_argv({
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=80x40:rate=2:duration=3",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        video_path.string(),
    });
    return video_path;
}

/**
 * @brief Exercises decoder metadata, frame ownership, and JPEG output bounds.
 *
 * @return Zero when all FFmpeg integration assertions pass.
 */
int run_ffmpeg_tests() {
    using test_support::Dimensions;
    using test_support::make_temp_directory;
    using test_support::read_jpeg_dimensions;
    using test_support::require_throws;
    using test_support::require_true;

    const std::filesystem::path directory = make_temp_directory("hcmai-ffmpeg");
    const std::filesystem::path video_path = make_synthetic_video(directory);
    const std::filesystem::path output_path = directory / "durable.jpg";

    VideoDecoder decoder(video_path);
    const VideoInfo info = decoder.info();
    require_true(
        info.avg_fps > 1.9 && info.avg_fps < 2.1,
        "average FPS must be read from stream"
    );
    require_true(info.width == 80 && info.height == 40, "source dimensions");
    require_true(
        info.avg_fps_rational.numerator > 0 &&
            info.avg_fps_rational.denominator > 0,
        "raw average FPS rational must be retained"
    );
    require_true(
        info.time_base.numerator > 0 && info.time_base.denominator > 0,
        "time base rational must be retained"
    );
    require_true(info.duration_ms >= 0, "stream duration must be retained");

    const auto first = decoder.next();
    require_true(first.has_value(), "synthetic video must decode");
    require_true(first->image != nullptr, "decoded image must be retained");
    require_true(
        first->timestamp_ms >= 0,
        "best effort timestamp must be converted"
    );
    require_true(first->pts >= 0, "best effort PTS must be retained");

    const auto second = decoder.next();
    require_true(second.has_value(), "second frame must decode");

    const EncodedImage encoded = encode_jpeg(
        *first->image,
        output_path,
        ImageVariant{32, 92}
    );
    require_true(encoded.bytes > 0, "JPEG must contain bytes");
    require_true(
        encoded.bytes == test_support::file_size(output_path),
        "reported JPEG bytes"
    );
    require_true(
        read_jpeg_dimensions(output_path) == Dimensions{32, 16},
        "long edge must be bounded"
    );

    const std::filesystem::path original_size_path = directory / "original.jpg";
    encode_jpeg(*first->image, original_size_path, ImageVariant{0, 92});
    require_true(
        read_jpeg_dimensions(original_size_path) == Dimensions{80, 40},
        "zero max long edge preserves source dimensions"
    );

    require_throws([&] {
        encode_jpeg(*first->image, directory / "invalid.jpg", ImageVariant{32, 0});
    });
    require_throws([&] {
        encode_jpeg(
            *first->image,
            directory / "missing-parent" / "invalid.jpg",
            ImageVariant{32, 92}
        );
    });
    require_throws([&] { VideoDecoder(directory / "missing.mp4"); });

    return test_support::finish_test();
}

}  // namespace
}  // namespace hcmai::keyframes_extraction

/**
 * @brief Runs the FFmpeg integration test executable.
 *
 * @param argc Number of command-line arguments; unused.
 * @param argv Command-line arguments; unused.
 * @return Zero when all assertions pass.
 */
int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    return hcmai::keyframes_extraction::run_ffmpeg_tests();
}
