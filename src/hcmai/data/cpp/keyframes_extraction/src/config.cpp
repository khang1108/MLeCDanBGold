#include "hcmai/keyframes_extraction/config.hpp"

#include <json-c/json.h>
#include <json-c/json_tokener.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hcmai::keyframes_extraction {
namespace {

class JsonObject {
public:
    explicit JsonObject(json_object* value) : value_(value) {}

    ~JsonObject() {
        if (value_ != nullptr) {
            json_object_put(value_);
        }
    }

    JsonObject(const JsonObject&) = delete;
    JsonObject& operator=(const JsonObject&) = delete;

    JsonObject(JsonObject&& other) noexcept : value_(other.value_) {
        other.value_ = nullptr;
    }

    JsonObject& operator=(JsonObject&& other) noexcept {
        if (this != &other) {
            if (value_ != nullptr) {
                json_object_put(value_);
            }
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }

    json_object* get() const {
        return value_;
    }

private:
    json_object* value_;
};

JsonObject parse_object(
    const std::filesystem::path& path,
    const std::string& text
) {
    json_tokener_error error = json_tokener_success;
    json_object* parsed = json_tokener_parse_verbose(text.c_str(), &error);
    if (parsed == nullptr || error != json_tokener_success) {
        throw std::invalid_argument(
            "invalid JSON configuration " + path.string() + ": " +
            json_tokener_error_desc(error)
        );
    }
    if (!json_object_is_type(parsed, json_type_object)) {
        json_object_put(parsed);
        throw std::invalid_argument(
            "configuration root must be a JSON object: " + path.string()
        );
    }
    return JsonObject(parsed);
}

json_object* required_member(
    json_object* object,
    const char* key,
    const std::filesystem::path& path
) {
    json_object* value = nullptr;
    if (!json_object_object_get_ex(object, key, &value) || value == nullptr) {
        throw std::invalid_argument(
            "missing configuration field '" + std::string(key) + "': " +
            path.string()
        );
    }
    return value;
}

std::int64_t required_integer(
    json_object* object,
    const char* key,
    const std::filesystem::path& path
) {
    json_object* value = required_member(object, key, path);
    if (!json_object_is_type(value, json_type_int)) {
        throw std::invalid_argument(
            "configuration field must be an integer '" + std::string(key) +
            "': " + path.string()
        );
    }
    return json_object_get_int64(value);
}

bool required_boolean(
    json_object* object,
    const char* key,
    const std::filesystem::path& path
) {
    json_object* value = required_member(object, key, path);
    if (!json_object_is_type(value, json_type_boolean)) {
        throw std::invalid_argument(
            "configuration field must be a boolean '" + std::string(key) +
            "': " + path.string()
        );
    }
    return json_object_get_boolean(value) != 0;
}

std::string required_string(
    json_object* object,
    const char* key,
    const std::filesystem::path& path
) {
    json_object* value = required_member(object, key, path);
    if (!json_object_is_type(value, json_type_string)) {
        throw std::invalid_argument(
            "configuration field must be a string '" + std::string(key) +
            "': " + path.string()
        );
    }
    const char* raw = json_object_get_string(value);
    if (raw == nullptr || std::string_view(raw).empty()) {
        throw std::invalid_argument(
            "configuration field must not be blank '" + std::string(key) +
            "': " + path.string()
        );
    }
    return raw;
}

int required_int(
    json_object* object,
    const char* key,
    const std::filesystem::path& path
) {
    const std::int64_t value = required_integer(object, key, path);
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "configuration integer is outside int range '" +
            std::string(key) + "': " + path.string()
        );
    }
    return static_cast<int>(value);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open JSON configuration: " + path.string()
        );
    }
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    );
}

}  // namespace

void validate_extraction_config(const ExtractionConfig& config) {
    if (config.sample_period_ms <= 0) {
        throw std::invalid_argument("sample_period_ms must be positive");
    }
    if (config.durable_long_edge <= 0) {
        throw std::invalid_argument("durable_long_edge must be positive");
    }
    if (config.durable_jpeg_quality < 1 ||
        config.durable_jpeg_quality > 100) {
        throw std::invalid_argument(
            "durable_jpeg_quality must be between 1 and 100"
        );
    }
    if (config.enrichment_jpeg_quality < 1 ||
        config.enrichment_jpeg_quality > 100) {
        throw std::invalid_argument(
            "enrichment_jpeg_quality must be between 1 and 100"
        );
    }
    if (config.yt_dlp_binary.empty()) {
        throw std::invalid_argument("yt_dlp_binary must not be blank");
    }
    if (config.extractor_version.empty()) {
        throw std::invalid_argument("extractor_version must not be blank");
    }
    if (config.config_hash.empty()) {
        throw std::invalid_argument("config_hash must not be blank");
    }
    if (!std::isfinite(static_cast<double>(config.sample_period_ms))) {
        throw std::invalid_argument("sample_period_ms must be finite");
    }
}

ExtractionConfig read_extraction_config(const std::filesystem::path& path) {
    const JsonObject root = parse_object(path, read_text(path));
    ExtractionConfig config;
    config.sample_period_ms = required_integer(root.get(), "sample_period_ms", path);
    config.durable_long_edge = required_int(
        root.get(), "durable_long_edge", path
    );
    config.durable_jpeg_quality = required_int(
        root.get(), "durable_jpeg_quality", path
    );
    config.enrichment_jpeg_quality = required_int(
        root.get(), "enrichment_jpeg_quality", path
    );
    config.write_enrichment_images = required_boolean(
        root.get(),
        "write_enrichment_images",
        path
    );
    config.yt_dlp_binary = required_string(root.get(), "yt_dlp_binary", path);
    config.extractor_version = required_string(
        root.get(),
        "extractor_version",
        path
    );
    config.config_hash = required_string(root.get(), "config_hash", path);
    validate_extraction_config(config);
    return config;
}

}  // namespace hcmai::keyframes_extraction
