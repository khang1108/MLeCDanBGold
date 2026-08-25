/**
 * @file process.hpp
 * @brief Declares shell-free POSIX subprocess execution for native extraction.
 *
 * This contract runs an explicit argument vector and captures bounded output.
 * It does not construct yt-dlp arguments, retry failed commands, or mutate
 * extraction state.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace hcmai::keyframes_extraction {

/**
 * @brief Maximum number of bytes retained for each captured output stream.
 *
 * The runner continues draining data after this limit so a verbose child cannot
 * block on a full pipe, but excess bytes are intentionally discarded.
 */
inline constexpr std::size_t kMaxCapturedProcessOutputBytes = 64U * 1024U;

/**
 * @brief Captures one completed child process outcome.
 */
struct ProcessResult {
    /** @brief Normal exit status, or -1 when the child ended by signal. */
    int exit_code = -1;
    /** @brief POSIX signal number, or zero when the child exited normally. */
    int signal_number = 0;
    /** @brief Bounded bytes written to the child's standard output stream. */
    std::string stdout_text;
    /** @brief Bounded bytes written to the child's standard error stream. */
    std::string stderr_text;
};

/**
 * @brief Executes a POSIX program using argv directly, without a shell.
 *
 * Each vector element becomes exactly one child argument. The function drains
 * stdout and stderr concurrently through pipes, retaining at most
 * kMaxCapturedProcessOutputBytes from each stream.
 *
 * @param argv Program path followed by its literal argument values.
 * @return The child's normal exit code or terminating signal and captured output.
 * @throws std::invalid_argument If argv is empty, argv[0] is blank, or an
 *                               argument contains an embedded NUL byte.
 * @throws std::system_error If pipe, fork, poll, read, or waitpid fails.
 */
ProcessResult run_process(const std::vector<std::string>& argv);

}  // namespace hcmai::keyframes_extraction
