/**
 * @file config.cpp
 * @brief Parses and validates native extractor configuration documents.
 *
 * This module owns json-c configuration parsing and semantic validation. It
 * does not execute extraction commands or write native frame artifacts.
 */

#include "hcmai/keyframes_extraction/config.hpp"

#include <json-c/json.h>
#include <json-c/json_tokener.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hcmai::keyframes_extraction
{
    namespace
    {

        /**
         * @brief Owns one json-c object using RAII reference release.
         *
         * The wrapper is move-only so every successfully parsed JSON object is
         * released exactly once on normal or exceptional paths.
         */
        class JsonObject
        {
        public:
            /**
             * @brief Takes ownership of a json-c object reference.
             *
             * @param value Owned json-c object pointer, possibly null.
             * @return None; constructors do not return values.
             */
            explicit JsonObject(json_object *value) : value_(value) {}

            /**
             * @brief Releases the owned json-c object when present.
             *
             * @return None; destructors do not return values.
             */
            ~JsonObject()
            {
                if (value_ != nullptr)
                {
                    json_object_put(value_);
                }
            }

            JsonObject(const JsonObject &) = delete;
            JsonObject &operator=(const JsonObject &) = delete;

            /**
             * @brief Transfers json-c ownership from another wrapper.
             *
             * @param other Wrapper whose ownership is transferred.
             * @return None; constructors do not return values.
             */
            JsonObject(JsonObject &&other) noexcept : value_(other.value_)
            {
                other.value_ = nullptr;
            }

            /**
             * @brief Replaces this wrapper's object by moving another wrapper.
             *
             * @param other Wrapper whose ownership is transferred.
             * @return This wrapper after ownership transfer.
             */
            JsonObject &operator=(JsonObject &&other) noexcept
            {
                if (this != &other)
                {
                    if (value_ != nullptr)
                    {
                        json_object_put(value_);
                    }
                    value_ = other.value_;
                    other.value_ = nullptr;
                }
                return *this;
            }

            /**
             * @brief Provides non-owning access to the wrapped json-c object.
             *
             * @return Raw json-c object pointer, which may be null.
             */
            json_object *get() const { return value_; }

        private:
            json_object *value_;
        };

        /**
         * @brief Parses a JSON configuration document whose root must be an object.
         *
         * @param path Source path used in validation errors.
         * @param text Complete UTF-8 JSON document content.
         * @return Owning wrapper for the parsed root object.
         * @throws std::invalid_argument If text is invalid JSON or not an object.
         */
        JsonObject parse_object(const std::filesystem::path &path,
                                const std::string &text)
        {
            json_tokener_error error = json_tokener_success;
            json_object *parsed = json_tokener_parse_verbose(text.c_str(), &error);

            if (parsed == nullptr || error != json_tokener_success)
            {
                throw std::invalid_argument("invalid JSON configuration " + path.string() +
                                            ": " + json_tokener_error_desc(error));
            }

            if (!json_object_is_type(parsed, json_type_object))
            {
                json_object_put(parsed);
                throw std::invalid_argument("configuration root must be a JSON object: " +
                                            path.string());
            }
            return JsonObject(parsed);
        }

        /**
         * @brief Retrieves a required configuration member without ownership transfer.
         *
         * @param object Parsed configuration root object.
         * @param key Required JSON member name.
         * @param path Source path used in validation errors.
         * @return Borrowed json-c pointer for the required member.
         * @throws std::invalid_argument If key is missing or resolves to null.
         */
        json_object *required_member(json_object *object, const char *key,
                                     const std::filesystem::path &path)
        {
            json_object *value = nullptr;
            if (!json_object_object_get_ex(object, key, &value) || value == nullptr)
            {
                throw std::invalid_argument("missing configuration field '" +
                                            std::string(key) + "': " + path.string());
            }
            return value;
        }

        /**
         * @brief Reads a required signed integer configuration member.
         *
         * @param object Parsed configuration root object.
         * @param key Required JSON member name.
         * @param path Source path used in validation errors.
         * @return The member value as int64_t.
         * @throws std::invalid_argument If key is missing or not an integer.
         */
        std::int64_t required_integer(json_object *object, const char *key,
                                      const std::filesystem::path &path)
        {
            json_object *value = required_member(object, key, path);
            if (!json_object_is_type(value, json_type_int))
            {
                throw std::invalid_argument("configuration field must be an integer '" +
                                            std::string(key) + "': " + path.string());
            }
            return json_object_get_int64(value);
        }

        /**
         * @brief Reads a required boolean configuration member.
         *
         * @param object Parsed configuration root object.
         * @param key Required JSON member name.
         * @param path Source path used in validation errors.
         * @return The member's boolean value.
         * @throws std::invalid_argument If key is missing or not a boolean.
         */
        bool required_boolean(json_object *object, const char *key,
                              const std::filesystem::path &path)
        {
            json_object *value = required_member(object, key, path);
            if (!json_object_is_type(value, json_type_boolean))
            {
                throw std::invalid_argument("configuration field must be a boolean '" +
                                            std::string(key) + "': " + path.string());
            }
            return json_object_get_boolean(value) != 0;
        }

        /**
         * @brief Reads a required non-blank string configuration member.
         *
         * @param object Parsed configuration root object.
         * @param key Required JSON member name.
         * @param path Source path used in validation errors.
         * @return A copied non-blank string value.
         * @throws std::invalid_argument If key is missing, not a string, or blank.
         */
        std::string required_string(json_object *object, const char *key,
                                    const std::filesystem::path &path)
        {
            json_object *value = required_member(object, key, path);
            if (!json_object_is_type(value, json_type_string))
            {
                throw std::invalid_argument("configuration field must be a string '" +
                                            std::string(key) + "': " + path.string());
            }
            const char *raw = json_object_get_string(value);
            if (raw == nullptr || std::string_view(raw).empty())
            {
                throw std::invalid_argument("configuration field must not be blank '" +
                                            std::string(key) + "': " + path.string());
            }
            return raw;
        }

        /**
         * @brief Reads an optional non-blank string, accepting absent or null.
         *
         * @param object Parsed configuration root object.
         * @param key Optional JSON member name.
         * @param path Source path used in validation errors.
         * @return A copied string when configured; otherwise nullopt.
         * @throws std::invalid_argument If a present value is not a non-blank string.
         */
        std::optional<std::string> optional_string(
            json_object *object,
            const char *key,
            const std::filesystem::path &path
        )
        {
            json_object *value = nullptr;
            if (!json_object_object_get_ex(object, key, &value) || value == nullptr ||
                json_object_is_type(value, json_type_null))
            {
                return std::nullopt;
            }
            if (!json_object_is_type(value, json_type_string))
            {
                throw std::invalid_argument("configuration field must be a string '" +
                                            std::string(key) + "': " + path.string());
            }
            const char *raw = json_object_get_string(value);
            if (raw == nullptr || std::string_view(raw).empty())
            {
                throw std::invalid_argument("configuration field must not be blank '" +
                                            std::string(key) + "': " + path.string());
            }
            return std::string(raw);
        }

        /**
         * @brief Reads a required integer that must fit in the native int type.
         *
         * @param object Parsed configuration root object.
         * @param key Required JSON member name.
         * @param path Source path used in validation errors.
         * @return The member value converted to int.
         * @throws std::invalid_argument If the value is absent, non-integer, or
         *                               outside the int range.
         */
        int required_int(json_object *object, const char *key,
                         const std::filesystem::path &path)
        {
            const std::int64_t value = required_integer(object, key, path);
            if (value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max())
            {
                throw std::invalid_argument("configuration integer is outside int range '" +
                                            std::string(key) + "': " + path.string());
            }
            return static_cast<int>(value);
        }

        /**
         * @brief Reads an optional integer member, defaulting when absent.
         *
         * @param object Parsed configuration root object.
         * @param key Optional JSON member name.
         * @param path Source path used in validation errors.
         * @param default_value Value returned when the member is absent or null.
         * @return The member value, or `default_value` when unset.
         * @throws std::invalid_argument If a present value is not an integer.
         */
        std::int64_t optional_integer(json_object *object, const char *key,
                                      const std::filesystem::path &path,
                                      std::int64_t default_value)
        {
            json_object *value = nullptr;
            if (!json_object_object_get_ex(object, key, &value) || value == nullptr ||
                json_object_is_type(value, json_type_null))
            {
                return default_value;
            }
            if (!json_object_is_type(value, json_type_int))
            {
                throw std::invalid_argument("configuration field must be an integer '" +
                                            std::string(key) + "': " + path.string());
            }
            return json_object_get_int64(value);
        }

        /**
         * @brief Reads a configuration file without changing its bytes.
         *
         * @param path Filesystem path to the configuration document.
         * @return Complete file content as a string.
         * @throws std::runtime_error If the file cannot be opened.
         */
        std::string read_text(const std::filesystem::path &path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("unable to open JSON configuration: " +
                                         path.string());
            }
            return std::string(std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>());
        }

    } // namespace

    /**
     * @brief Validates all values required for deterministic native extraction.
     *
     * @param config Candidate configuration to validate.
     * @return None; throws when a required invariant is violated.
     * @throws std::invalid_argument If a value is outside its supported range.
     */
    void validate_extraction_config(const ExtractionConfig &config)
    {
        if (config.sample_period_ms <= 0)
        {
            throw std::invalid_argument("sample_period_ms must be positive");
        }
        if (config.durable_long_edge <= 0)
        {
            throw std::invalid_argument("durable_long_edge must be positive");
        }
        if (config.durable_jpeg_quality < 1 || config.durable_jpeg_quality > 100)
        {
            throw std::invalid_argument(
                "durable_jpeg_quality must be between 1 and 100");
        }
        if (config.enrichment_jpeg_quality < 1 ||
            config.enrichment_jpeg_quality > 100)
        {
            throw std::invalid_argument(
                "enrichment_jpeg_quality must be between 1 and 100");
        }
        if (config.yt_dlp_binary.empty())
        {
            throw std::invalid_argument("yt_dlp_binary must not be blank");
        }
        if (config.disk_reserve_bytes < 0)
        {
            throw std::invalid_argument("disk_reserve_bytes must not be negative");
        }
        if (config.extractor_version.empty())
        {
            throw std::invalid_argument("extractor_version must not be blank");
        }
        if (config.config_hash.empty())
        {
            throw std::invalid_argument("config_hash must not be blank");
        }
        if (!std::isfinite(static_cast<double>(config.sample_period_ms)))
        {
            throw std::invalid_argument("sample_period_ms must be finite");
        }
    }

    /**
     * @brief Loads JSON configuration and applies semantic validation.
     *
     * @param path Filesystem path to the native extractor configuration JSON.
     * @return A validated ExtractionConfig ready for use by native stages.
     * @throws std::runtime_error If path cannot be read.
     * @throws std::invalid_argument If parsing or validation fails.
     */
    ExtractionConfig read_extraction_config(const std::filesystem::path &path)
    {
        const JsonObject root = parse_object(path, read_text(path));
        ExtractionConfig config;
        config.sample_period_ms =
            required_integer(root.get(), "sample_period_ms", path);
        config.durable_long_edge =
            required_int(root.get(), "durable_long_edge", path);
        config.durable_jpeg_quality =
            required_int(root.get(), "durable_jpeg_quality", path);
        config.enrichment_jpeg_quality =
            required_int(root.get(), "enrichment_jpeg_quality", path);
        config.write_enrichment_images =
            required_boolean(root.get(), "write_enrichment_images", path);
        config.disk_reserve_bytes =
            optional_integer(root.get(), "disk_reserve_bytes", path, 0);
        config.yt_dlp_binary = required_string(root.get(), "yt_dlp_binary", path);
        config.yt_dlp_cookies_path =
            optional_string(root.get(), "yt_dlp_cookies_path", path);
        config.yt_dlp_js_runtime =
            optional_string(root.get(), "yt_dlp_js_runtime", path);
        config.extractor_version =
            required_string(root.get(), "extractor_version", path);
        config.config_hash = required_string(root.get(), "config_hash", path);
        validate_extraction_config(config);
        return config;
    }

} // namespace hcmai::keyframes_extraction
