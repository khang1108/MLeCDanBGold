#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace hcmai::keyframes_extraction::test_support {

struct Dimensions {
    int width;
    int height;

    friend bool operator==(const Dimensions& left, const Dimensions& right) {
        return left.width == right.width && left.height == right.height;
    }
};

inline void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable&& callable) {
    bool threw = false;
    try {
        std::invoke(std::forward<Callable>(callable));
    } catch (...) {
        threw = true;
    }
    require_true(threw, "expected callable to throw");
}

inline std::filesystem::path make_temp_directory(std::string_view prefix) {
    static std::atomic<std::uint64_t> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();

    const auto root = std::filesystem::temp_directory_path();
    for (std::uint64_t attempt = 0; attempt < 1000; ++attempt) {
        const auto suffix = std::to_string(now) + "-" +
            std::to_string(counter.fetch_add(1)) + "-" +
            std::to_string(attempt);
        const auto directory = root / (std::string(prefix) + "-" + suffix);
        std::error_code error;
        if (std::filesystem::create_directory(directory, error)) {
            return directory;
        }
        if (error && error != std::errc::file_exists) {
            throw std::system_error(error, "create test directory");
        }
    }

    throw std::runtime_error("unable to create unique test directory");
}

inline void write_text(
    const std::filesystem::path& path,
    std::string_view contents
) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("unable to open test file for writing");
    }
    output << contents;
    if (!output) {
        throw std::runtime_error("unable to write test file");
    }
}

inline std::uintmax_t file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        throw std::system_error(error, "read test file size");
    }
    return size;
}

inline bool exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

inline bool is_regular_file(const std::filesystem::path& path) {
    return std::filesystem::is_regular_file(path);
}

inline Dimensions read_jpeg_dimensions(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open JPEG");
    }

    const auto read_byte = [&input]() -> int {
        return input.get();
    };
    if (read_byte() != 0xff || read_byte() != 0xd8) {
        throw std::runtime_error("not a JPEG stream");
    }

    while (input) {
        int marker_prefix = read_byte();
        while (marker_prefix == 0xff) {
            marker_prefix = read_byte();
        }
        if (marker_prefix < 0) {
            break;
        }
        if (marker_prefix == 0xd9 || marker_prefix == 0xda) {
            break;
        }
        const int length_high = read_byte();
        const int length_low = read_byte();
        if (length_high < 0 || length_low < 0) {
            break;
        }
        const int length = (length_high << 8) | length_low;
        if (length < 2) {
            throw std::runtime_error("invalid JPEG segment length");
        }

        const bool is_start_of_frame =
            (marker_prefix >= 0xc0 && marker_prefix <= 0xc3) ||
            (marker_prefix >= 0xc5 && marker_prefix <= 0xc7) ||
            (marker_prefix >= 0xc9 && marker_prefix <= 0xcb) ||
            (marker_prefix >= 0xcd && marker_prefix <= 0xcf);
        if (is_start_of_frame) {
            const int precision = read_byte();
            const int height_high = read_byte();
            const int height_low = read_byte();
            const int width_high = read_byte();
            const int width_low = read_byte();
            if (precision < 0 || height_high < 0 || height_low < 0 ||
                width_high < 0 || width_low < 0) {
                throw std::runtime_error("truncated JPEG dimensions");
            }
            return Dimensions{
                (width_high << 8) | width_low,
                (height_high << 8) | height_low,
            };
        }

        input.seekg(length - 2, std::ios::cur);
    }

    throw std::runtime_error("JPEG dimensions not found");
}

inline int finish_test() {
    return 0;
}

}  // namespace hcmai::keyframes_extraction::test_support
