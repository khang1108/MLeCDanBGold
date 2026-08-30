/**
 * @file test_process.cpp
 * @brief Verifies POSIX subprocess execution preserves explicit argv boundaries.
 *
 * These tests ensure untrusted watch URLs and other arguments cannot be
 * interpreted as shell syntax by the native extractor.
 */

#include "hcmai/keyframes_extraction/process.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <string>

/**
 * @brief Exercises successful, failed, and signaled explicit-argv processes.
 *
 * @return Zero when all process execution invariants hold; throws otherwise.
 */
int main() {
    using hcmai::keyframes_extraction::ProcessResult;
    using hcmai::keyframes_extraction::run_process;
    using namespace hcmai::keyframes_extraction::test_support;

    const std::filesystem::path root = make_temp_directory("process");
    const std::filesystem::path sentinel = root / "should-not-exist";
    const std::string literal =
        "literal;$(touch " + sentinel.string() + ")";

    const ProcessResult literal_result = run_process({
        "/usr/bin/printf", "%s", literal,
    });
    require_true(literal_result.exit_code == 0,
                 "literal argv process must succeed");
    require_true(literal_result.signal_number == 0,
                 "successful process must not report a signal");
    require_true(literal_result.stdout_text == literal,
                 "shell metacharacters must remain one literal argument");
    require_true(!hcmai::keyframes_extraction::test_support::exists(sentinel),
                 "argv execution must not evaluate command substitution");

    const ProcessResult missing_result = run_process({
        "keyframes-extractor-command-that-does-not-exist",
    });
    require_true(missing_result.exit_code == 127,
                 "exec failure must return the child failure code");
    require_true(!missing_result.stderr_text.empty(),
                 "exec failure must retain diagnostics in stderr");

    const ProcessResult bounded_result = run_process({
        "/bin/sh", "-c", "head -c 70000 /dev/zero >&2",
    });
    require_true(bounded_result.exit_code == 0,
                 "large stderr producer must complete without a blocked pipe");
    require_true(
        bounded_result.stderr_text.size() ==
            hcmai::keyframes_extraction::kMaxCapturedProcessOutputBytes,
        "stderr retention must be capped at the configured bound"
    );

    const ProcessResult signaled_result = run_process({
        "/bin/sh", "-c", "kill -TERM $$",
    });
    require_true(signaled_result.exit_code == -1,
                 "signaled process must not report a normal exit code");
    require_true(signaled_result.signal_number == 15,
                 "SIGTERM must be preserved in the result");

    return finish_test();
}
