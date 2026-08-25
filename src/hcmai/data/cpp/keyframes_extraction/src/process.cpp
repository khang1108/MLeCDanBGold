/**
 * @file process.cpp
 * @brief Implements shell-free POSIX subprocess execution with bounded output.
 *
 * This module owns fork/exec, stdout/stderr capture, and child status handling.
 * It intentionally does not choose downloader arguments, retry commands, or
 * update native extraction state.
 */

#include "hcmai/keyframes_extraction/process.hpp"

#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Owns one POSIX file descriptor and closes it at scope exit.
 *
 * The wrapper is move-only so every pipe endpoint is closed exactly once in
 * the parent process, including exceptional paths.
 */
class FileDescriptor {
public:
    /**
     * @brief Takes ownership of a POSIX file descriptor.
     *
     * @param descriptor Owned descriptor, or -1 when no descriptor is open.
     * @return None; constructors do not return values.
     */
    explicit FileDescriptor(int descriptor = -1) noexcept
        : descriptor_(descriptor) {}

    /**
     * @brief Closes the owned descriptor when it is still open.
     *
     * @return None; destructors do not return values.
     */
    ~FileDescriptor() {
        reset();
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    /**
     * @brief Transfers descriptor ownership from another wrapper.
     *
     * @param other Wrapper whose descriptor ownership is transferred.
     * @return None; constructors do not return values.
     */
    FileDescriptor(FileDescriptor&& other) noexcept
        : descriptor_(std::exchange(other.descriptor_, -1)) {}

    /**
     * @brief Replaces this descriptor by moving ownership from another wrapper.
     *
     * @param other Wrapper whose descriptor ownership is transferred.
     * @return This wrapper after ownership transfer.
     */
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset(std::exchange(other.descriptor_, -1));
        }
        return *this;
    }

    /**
     * @brief Returns the raw owned descriptor without transferring ownership.
     *
     * @return Open POSIX descriptor, or -1 when no descriptor is owned.
     */
    int get() const noexcept {
        return descriptor_;
    }

    /**
     * @brief Reports whether this wrapper currently owns an open descriptor.
     *
     * @return True when get() is non-negative; otherwise false.
     */
    bool valid() const noexcept {
        return descriptor_ >= 0;
    }

    /**
     * @brief Closes the old descriptor and optionally adopts a replacement.
     *
     * @param descriptor Replacement descriptor, or -1 to leave the wrapper empty.
     * @return None; close failures are intentionally ignored during cleanup.
     */
    void reset(int descriptor = -1) noexcept {
        if (descriptor_ >= 0) {
            static_cast<void>(::close(descriptor_));
        }
        descriptor_ = descriptor;
    }

private:
    int descriptor_;
};

/**
 * @brief Throws a system_error using a previously captured POSIX errno value.
 *
 * @param operation Human-readable operation name for the diagnostic message.
 * @param error_number POSIX errno captured immediately after the failure.
 * @return None; this function always throws.
 * @throws std::system_error Always, using error_number and generic_category().
 */
[[noreturn]] void throw_errno(std::string_view operation, int error_number) {
    throw std::system_error(
        error_number,
        std::generic_category(),
        std::string(operation)
    );
}

/**
 * @brief Determines whether a command path is empty or whitespace-only.
 *
 * @param value Candidate program path text.
 * @return True when value contains no non-whitespace character; otherwise false.
 */
bool is_blank(std::string_view value) {
    if (value.empty()) {
        return true;
    }

    for (const unsigned char character : value) {
        if (std::isspace(character) == 0) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Validates and converts immutable C++ arguments into execvp pointers.
 *
 * @param argv Program path followed by literal child arguments.
 * @return A null-terminated pointer vector valid while argv remains alive.
 * @throws std::invalid_argument If argv cannot be represented by execvp.
 */
std::vector<char*> make_exec_argv(const std::vector<std::string>& argv) {
    if (argv.empty() || is_blank(argv.front())) {
        throw std::invalid_argument("argv must contain a non-blank program path");
    }

    std::vector<char*> result;
    result.reserve(argv.size() + 1U);
    for (const std::string& argument : argv) {
        if (argument.find('\0') != std::string::npos) {
            throw std::invalid_argument("argv values must not contain NUL bytes");
        }
        result.push_back(const_cast<char*>(argument.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

/**
 * @brief Creates one unidirectional POSIX pipe.
 *
 * @param label Short pipe name included in system error diagnostics.
 * @return Read endpoint followed by write endpoint, both owned by the caller.
 * @throws std::system_error If pipe creation fails.
 */
std::array<int, 2> make_pipe(std::string_view label) {
    std::array<int, 2> descriptors{-1, -1};
    if (::pipe(descriptors.data()) != 0) {
        const int error_number = errno;
        throw_errno(std::string(label) + " pipe", error_number);
    }
    return descriptors;
}

/**
 * @brief Writes a minimal diagnostic and terminates the child after setup failure.
 *
 * @param message Static diagnostic text safe to write after fork.
 * @param length Number of bytes from message to write.
 * @return None; this function terminates the child with exit code 127.
 */
[[noreturn]] void child_fail(const char* message, std::size_t length) noexcept {
    static_cast<void>(::write(STDERR_FILENO, message, length));
    _exit(127);
}

/**
 * @brief Rebinds child stdout/stderr to capture pipes and executes argv.
 *
 * @param stdout_read Parent-only read endpoint of the stdout capture pipe.
 * @param stdout_write Child write endpoint of the stdout capture pipe.
 * @param stderr_read Parent-only read endpoint of the stderr capture pipe.
 * @param stderr_write Child write endpoint of the stderr capture pipe.
 * @param argv Null-terminated execvp argument pointer array.
 * @return None; successful exec never returns and failures exit the child.
 */
[[noreturn]] void execute_child(
    int stdout_read,
    int stdout_write,
    int stderr_read,
    int stderr_write,
    char* const argv[]
) noexcept {
    static_cast<void>(::close(stdout_read));
    static_cast<void>(::close(stderr_read));

    if (::dup2(stdout_write, STDOUT_FILENO) == -1) {
        constexpr char message[] = "unable to redirect child stdout\n";
        child_fail(message, sizeof(message) - 1U);
    }
    if (::dup2(stderr_write, STDERR_FILENO) == -1) {
        constexpr char message[] = "unable to redirect child stderr\n";
        child_fail(message, sizeof(message) - 1U);
    }

    static_cast<void>(::close(stdout_write));
    static_cast<void>(::close(stderr_write));
    ::execvp(argv[0], argv);

    constexpr char message[] = "execvp failed\n";
    child_fail(message, sizeof(message) - 1U);
}

/**
 * @brief Appends bytes without exceeding the configured capture limit.
 *
 * @param output Mutable captured output destination.
 * @param data Source byte array read from a pipe.
 * @param size Number of source bytes available in data.
 * @return None; excess bytes are intentionally discarded after the limit.
 */
void append_bounded(std::string& output, const char* data, std::size_t size) {
    if (output.size() >= kMaxCapturedProcessOutputBytes) {
        return;
    }

    const std::size_t available = kMaxCapturedProcessOutputBytes - output.size();
    output.append(data, size < available ? size : available);
}

/**
 * @brief Drains one ready pipe endpoint into a bounded output string.
 *
 * @param descriptor Ready pipe descriptor owned by the parent process.
 * @param output Captured output destination for bytes read from descriptor.
 * @return None; closes descriptor after end-of-file.
 * @throws std::system_error If read fails for a non-retryable reason.
 */
void drain_ready_pipe(FileDescriptor& descriptor, std::string& output) {
    std::array<char, 8192> buffer{};
    const ssize_t byte_count = ::read(
        descriptor.get(),
        buffer.data(),
        buffer.size()
    );
    if (byte_count > 0) {
        append_bounded(
            output,
            buffer.data(),
            static_cast<std::size_t>(byte_count)
        );
        return;
    }
    if (byte_count == 0) {
        descriptor.reset();
        return;
    }
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        return;
    }

    const int error_number = errno;
    throw_errno("read child output", error_number);
}

/**
 * @brief Drains one pipe when poll reports a readable, closed, or error event.
 *
 * @param events Poll event bit set for descriptor.
 * @param descriptor Parent-owned pipe descriptor associated with events.
 * @param output Bounded capture destination for bytes read from descriptor.
 * @return None; drains readable data or raises on an invalid descriptor.
 * @throws std::system_error If poll reports an invalid descriptor or reading fails.
 */
void drain_poll_events(
    short events,
    FileDescriptor& descriptor,
    std::string& output
) {
    if ((events & POLLNVAL) != 0) {
        throw std::system_error(
            EBADF,
            std::generic_category(),
            "poll child output"
        );
    }
    if ((events & (POLLIN | POLLHUP | POLLERR)) != 0) {
        drain_ready_pipe(descriptor, output);
    }
}

/**
 * @brief Drains stdout and stderr until the child closes both capture pipes.
 *
 * @param stdout_descriptor Parent read endpoint for child standard output.
 * @param stderr_descriptor Parent read endpoint for child standard error.
 * @param result Mutable process result that receives captured output.
 * @return None; both descriptors are closed after their end-of-file markers.
 * @throws std::system_error If poll or pipe reading fails.
 */
void collect_child_output(
    FileDescriptor& stdout_descriptor,
    FileDescriptor& stderr_descriptor,
    ProcessResult& result
) {
    while (stdout_descriptor.valid() || stderr_descriptor.valid()) {
        std::array<pollfd, 2> descriptors{{
            {stdout_descriptor.get(), POLLIN, 0},
            {stderr_descriptor.get(), POLLIN, 0},
        }};

        int poll_result = 0;
        do {
            poll_result = ::poll(
                descriptors.data(),
                static_cast<nfds_t>(descriptors.size()),
                -1
            );
        } while (poll_result == -1 && errno == EINTR);

        if (poll_result == -1) {
            const int error_number = errno;
            throw_errno("poll child output", error_number);
        }

        drain_poll_events(
            descriptors[0].revents,
            stdout_descriptor,
            result.stdout_text
        );
        drain_poll_events(
            descriptors[1].revents,
            stderr_descriptor,
            result.stderr_text
        );
    }
}

/**
 * @brief Waits for a child process to reach a terminal status.
 *
 * @param child Process identifier returned by fork().
 * @return Raw POSIX wait status for the completed child.
 * @throws std::system_error If waitpid fails for a non-interrupt reason.
 */
int wait_for_child(pid_t child) {
    int status = 0;
    while (::waitpid(child, &status, 0) == -1) {
        if (errno != EINTR) {
            const int error_number = errno;
            throw_errno("waitpid", error_number);
        }
    }
    return status;
}

/**
 * @brief Reaps a child without throwing while unwinding a parent-side failure.
 *
 * @param child Process identifier returned by fork().
 * @return None; failures are suppressed to preserve the original exception.
 */
void reap_child_noexcept(pid_t child) noexcept {
    int status = 0;
    while (::waitpid(child, &status, 0) == -1 && errno == EINTR) {
    }
}

/**
 * @brief Converts a raw POSIX wait status into the public process result fields.
 *
 * @param status Raw status returned by waitpid().
 * @param result Mutable result that receives exit or signal information.
 * @return None; result is updated for normal exit or signal termination.
 * @throws std::runtime_error If status is not a terminal child result.
 */
void populate_terminal_status(int status, ProcessResult& result) {
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
        result.signal_number = 0;
        return;
    }
    if (WIFSIGNALED(status)) {
        result.exit_code = -1;
        result.signal_number = WTERMSIG(status);
        return;
    }

    throw std::runtime_error("child did not reach a terminal wait status");
}

}  // namespace

/**
 * @brief Executes a program with literal argv values and captures bounded output.
 *
 * @param argv Program path followed by literal child argument values.
 * @return Child completion status plus bounded stdout and stderr text.
 * @throws std::invalid_argument If argv cannot be passed safely to execvp.
 * @throws std::system_error If the POSIX process or capture lifecycle fails.
 */
ProcessResult run_process(const std::vector<std::string>& argv) {
    std::vector<char*> child_argv = make_exec_argv(argv);
    const std::array<int, 2> stdout_pipe = make_pipe("stdout");
    const std::array<int, 2> stderr_pipe = make_pipe("stderr");

    FileDescriptor stdout_read(stdout_pipe[0]);
    FileDescriptor stdout_write(stdout_pipe[1]);
    FileDescriptor stderr_read(stderr_pipe[0]);
    FileDescriptor stderr_write(stderr_pipe[1]);

    const pid_t child = ::fork();
    if (child == -1) {
        const int error_number = errno;
        throw_errno("fork", error_number);
    }
    if (child == 0) {
        execute_child(
            stdout_read.get(),
            stdout_write.get(),
            stderr_read.get(),
            stderr_write.get(),
            child_argv.data()
        );
    }

    stdout_write.reset();
    stderr_write.reset();

    ProcessResult result;
    try {
        collect_child_output(stdout_read, stderr_read, result);
        populate_terminal_status(wait_for_child(child), result);
    } catch (...) {
        stdout_read.reset();
        stderr_read.reset();
        reap_child_noexcept(child);
        throw;
    }

    return result;
}

}  // namespace hcmai::keyframes_extraction
