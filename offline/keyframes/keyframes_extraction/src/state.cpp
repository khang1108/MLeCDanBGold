/**
 * @file state.cpp
 * @brief Implements atomic JSON video checkpoints and guarded lifecycle changes.
 *
 * This module owns native state serialization, provenance validation, guarded
 * enrichment handoff acceptance, atomic bundle publication, and scoped cleanup.
 * It does not download sources or run any enrichment model.
 */

#include "hcmai/keyframes_extraction/state.hpp"

#include "hcmai/keyframes_extraction/frame_index.hpp"

#include <json-c/json.h>
#include <json-c/json_tokener.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Owns one json-c reference and releases it through RAII.
 *
 * The wrapper is move-only so parsed and serialized JSON objects do not leak
 * when validation or filesystem operations throw.
 */
class JsonObject {
public:
    /**
     * @brief Takes ownership of one json-c object reference.
     *
     * @param value Owned json-c pointer, which must not be null.
     * @return None; constructors do not return values.
     * @throws std::bad_alloc If value is null because json-c allocation failed.
     */
    explicit JsonObject(json_object* value) : value_(value) {
        if (value_ == nullptr) {
            throw std::bad_alloc();
        }
    }

    /**
     * @brief Releases the owned json-c reference.
     *
     * @return None; destructors do not return values.
     */
    ~JsonObject() {
        if (value_ != nullptr) {
            json_object_put(value_);
        }
    }

    JsonObject(const JsonObject&) = delete;
    JsonObject& operator=(const JsonObject&) = delete;

    /**
     * @brief Transfers json-c ownership from another wrapper.
     *
     * @param other Wrapper whose json-c reference is transferred.
     * @return None; constructors do not return values.
     */
    JsonObject(JsonObject&& other) noexcept
        : value_(std::exchange(other.value_, nullptr)) {}

    /**
     * @brief Replaces this JSON reference by moving ownership from another wrapper.
     *
     * @param other Wrapper whose json-c reference is transferred.
     * @return This wrapper after ownership transfer.
     */
    JsonObject& operator=(JsonObject&& other) noexcept {
        if (this != &other) {
            if (value_ != nullptr) {
                json_object_put(value_);
            }
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }

    /**
     * @brief Returns the raw json-c pointer without transferring ownership.
     *
     * @return Owned non-null json-c object pointer.
     */
    json_object* get() const noexcept {
        return value_;
    }

private:
    json_object* value_;
};

/**
 * @brief Tests whether text is empty or consists only of whitespace characters.
 *
 * @param value Text value to inspect.
 * @return True when value is blank; otherwise false.
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
 * @brief Rejects a required state field that is blank or whitespace-only.
 *
 * @param value Candidate string value.
 * @param field_name Field name included in the exception diagnostic.
 * @return None; returns only when value is non-blank.
 * @throws std::invalid_argument If value is blank.
 */
void require_non_blank(std::string_view value, std::string_view field_name) {
    if (is_blank(value)) {
        throw std::invalid_argument(
            "state field must not be blank: " + std::string(field_name)
        );
    }
}

/**
 * @brief Validates the immutable values required to own a state file.
 *
 * @param identity Candidate run, video, version, and configuration identity.
 * @return None; returns only when every identity field is non-blank.
 * @throws std::invalid_argument If any provenance field is blank.
 */
void validate_identity(const StateIdentity& identity) {
    require_non_blank(identity.run_id, "run_id");
    require_non_blank(identity.video_id, "video_id");
    require_non_blank(identity.extractor_version, "extractor_version");
    require_non_blank(identity.config_hash, "config_hash");
}

/**
 * @brief Converts a status enum into its stable JSON lifecycle token.
 *
 * @param status Native lifecycle status to serialize.
 * @return Lowercase stable token used in the state JSON contract.
 */
std::string_view status_name(VideoStatus status) {
    switch (status) {
    case VideoStatus::Pending:
        return "pending";
    case VideoStatus::Downloading:
        return "downloading";
    case VideoStatus::Extracting:
        return "extracting";
    case VideoStatus::Extracted:
        return "extracted";
    case VideoStatus::EnrichmentPending:
        return "enrichment_pending";
    case VideoStatus::Enriched:
        return "enriched";
    case VideoStatus::Published:
        return "published";
    case VideoStatus::Cleaned:
        return "cleaned";
    case VideoStatus::Failed:
        return "failed";
    }

    throw std::invalid_argument("unknown video status");
}

/**
 * @brief Parses a stable JSON lifecycle token into its status enum.
 *
 * @param value Lowercase status token read from a state document.
 * @return Matching VideoStatus value.
 * @throws std::invalid_argument If value is not a recognized lifecycle token.
 */
VideoStatus parse_status(std::string_view value) {
    if (value == "pending") {
        return VideoStatus::Pending;
    }
    if (value == "downloading") {
        return VideoStatus::Downloading;
    }
    if (value == "extracting") {
        return VideoStatus::Extracting;
    }
    if (value == "extracted") {
        return VideoStatus::Extracted;
    }
    if (value == "enrichment_pending") {
        return VideoStatus::EnrichmentPending;
    }
    if (value == "enriched") {
        return VideoStatus::Enriched;
    }
    if (value == "published") {
        return VideoStatus::Published;
    }
    if (value == "cleaned") {
        return VideoStatus::Cleaned;
    }
    if (value == "failed") {
        return VideoStatus::Failed;
    }

    throw std::invalid_argument("unknown video status: " + std::string(value));
}

/**
 * @brief Ensures a uint64 counter can be represented by json-c's signed integer API.
 *
 * @param value Non-negative native counter to serialize.
 * @param field_name Field name included in an overflow diagnostic.
 * @return None; returns only when value fits in int64_t.
 * @throws std::overflow_error If value exceeds int64_t maximum.
 */
void validate_json_integer_range(std::uint64_t value, std::string_view field_name) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "state field exceeds JSON integer range: " + std::string(field_name)
        );
    }
}

/**
 * @brief Validates all serializable invariants of a native video state.
 *
 * @param state Candidate state value to validate.
 * @return None; returns only when the state is internally consistent.
 * @throws std::invalid_argument If required values or error/status semantics fail.
 * @throws std::overflow_error If a counter cannot be written through json-c.
 */
void validate_state(const VideoState& state) {
    validate_identity(StateIdentity{
        state.run_id,
        state.video_id,
        state.extractor_version,
        state.config_hash,
    });
    require_non_blank(state.watch_url, "watch_url");
    require_non_blank(state.started_at, "started_at");
    require_non_blank(state.updated_at, "updated_at");
    validate_json_integer_range(state.emitted_frame_count, "emitted_frame_count");
    if (state.last_completed_sample_index.has_value()) {
        validate_json_integer_range(
            state.last_completed_sample_index.value(),
            "last_completed_sample_index"
        );
    }

    if (state.status == VideoStatus::Failed) {
        require_non_blank(state.error, "error");
        if (state.error.size() > kMaxStoredStateErrorBytes) {
            throw std::invalid_argument("failed state error exceeds capture bound");
        }
        return;
    }

    if (!state.error.empty()) {
        throw std::invalid_argument(
            "only failed state may retain an error diagnostic"
        );
    }
}

/**
 * @brief Produces a bounded copy of a failure diagnostic for durable state JSON.
 *
 * @param error Source process or decoder diagnostic text.
 * @return At most kMaxStoredStateErrorBytes from error, preserving original byte order.
 */
std::string bounded_error(std::string error) {
    if (error.size() > kMaxStoredStateErrorBytes) {
        error.resize(kMaxStoredStateErrorBytes);
    }
    return error;
}

/**
 * @brief Gets the current UTC time in the stable state JSON timestamp format.
 *
 * @return ISO-8601 UTC timestamp with second precision.
 * @throws std::runtime_error If the platform cannot convert system time to UTC.
 */
std::string utc_timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now()
    );
    std::tm utc{};
    if (::gmtime_r(&now, &utc) == nullptr) {
        throw std::runtime_error("unable to convert state timestamp to UTC");
    }

    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

/**
 * @brief Reads a complete state JSON file without changing its content.
 *
 * @param state_path Path to the JSON state document.
 * @return Entire file byte sequence as a string.
 * @throws std::runtime_error If state_path cannot be opened.
 */
std::string read_text(const std::filesystem::path& state_path) {
    std::ifstream input(state_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to open state file: " + state_path.string()
        );
    }

    std::ostringstream content;
    content << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error(
            "unable to read state file: " + state_path.string()
        );
    }
    return content.str();
}

/**
 * @brief Parses a JSON document and requires an object root for native state.
 *
 * @param state_path Source path used in validation error messages.
 * @param text Complete state JSON content.
 * @return Owning json-c wrapper for the parsed root object.
 * @throws std::invalid_argument If text is invalid JSON or not an object.
 */
JsonObject parse_object(
    const std::filesystem::path& state_path,
    const std::string& text
) {
    json_tokener_error error = json_tokener_success;
    json_object* parsed = json_tokener_parse_verbose(text.c_str(), &error);
    if (parsed == nullptr || error != json_tokener_success) {
        if (parsed != nullptr) {
            json_object_put(parsed);
        }
        throw std::invalid_argument(
            "invalid state JSON " + state_path.string() + ": " +
            json_tokener_error_desc(error)
        );
    }
    if (!json_object_is_type(parsed, json_type_object)) {
        json_object_put(parsed);
        throw std::invalid_argument(
            "state root must be a JSON object: " + state_path.string()
        );
    }
    return JsonObject(parsed);
}

/**
 * @brief Retrieves one required state JSON member without ownership transfer.
 *
 * @param object Parsed state root object.
 * @param key Required top-level state field name.
 * @param state_path Source path used in validation error messages.
 * @return Borrowed json-c pointer for the requested member; JSON null is
 *         represented by a null pointer in json-c.
 * @throws std::invalid_argument If key is absent.
 */
json_object* required_member(
    json_object* object,
    const char* key,
    const std::filesystem::path& state_path
) {
    json_object* value = nullptr;
    if (!json_object_object_get_ex(object, key, &value)) {
        throw std::invalid_argument(
            "missing state field '" + std::string(key) + "': " +
            state_path.string()
        );
    }
    return value;
}

/**
 * @brief Reads one required JSON string field while preserving its byte content.
 *
 * @param object Parsed state root object.
 * @param key Required state field name.
 * @param state_path Source path used in validation error messages.
 * @return Copied string field value, which may be empty for optional paths.
 * @throws std::invalid_argument If the field is absent or not a JSON string.
 */
std::string required_string(
    json_object* object,
    const char* key,
    const std::filesystem::path& state_path
) {
    json_object* value = required_member(object, key, state_path);
    if (!json_object_is_type(value, json_type_string)) {
        throw std::invalid_argument(
            "state field must be a string '" + std::string(key) + "': " +
            state_path.string()
        );
    }

    const char* raw = json_object_get_string(value);
    if (raw == nullptr) {
        return std::string();
    }
    const int length = json_object_get_string_len(value);
    return std::string(raw, static_cast<std::size_t>(length));
}

/**
 * @brief Reads one required non-negative JSON integer into a uint64 value.
 *
 * @param object Parsed state root object.
 * @param key Required state field name.
 * @param state_path Source path used in validation error messages.
 * @return Non-negative integer field represented as uint64_t.
 * @throws std::invalid_argument If the field is absent, non-integer, or negative.
 */
std::uint64_t required_u64(
    json_object* object,
    const char* key,
    const std::filesystem::path& state_path
) {
    json_object* value = required_member(object, key, state_path);
    if (!json_object_is_type(value, json_type_int)) {
        throw std::invalid_argument(
            "state field must be an integer '" + std::string(key) + "': " +
            state_path.string()
        );
    }

    const std::int64_t integer = json_object_get_int64(value);
    if (integer < 0) {
        throw std::invalid_argument(
            "state field must be non-negative '" + std::string(key) + "': " +
            state_path.string()
        );
    }
    return static_cast<std::uint64_t>(integer);
}

/**
 * @brief Reads a nullable non-negative JSON integer field.
 *
 * @param object Parsed state root object.
 * @param key Required nullable state field name.
 * @param state_path Source path used in validation error messages.
 * @return Nullopt for JSON null; otherwise the non-negative integer value.
 * @throws std::invalid_argument If the field is absent or neither null nor integer.
 */
std::optional<std::uint64_t> required_optional_u64(
    json_object* object,
    const char* key,
    const std::filesystem::path& state_path
) {
    json_object* value = required_member(object, key, state_path);
    if (value == nullptr || json_object_is_type(value, json_type_null)) {
        return std::nullopt;
    }
    if (!json_object_is_type(value, json_type_int)) {
        throw std::invalid_argument(
            "state field must be null or integer '" + std::string(key) + "': " +
            state_path.string()
        );
    }

    const std::int64_t integer = json_object_get_int64(value);
    if (integer < 0) {
        throw std::invalid_argument(
            "state field must be non-negative '" + std::string(key) + "': " +
            state_path.string()
        );
    }
    return static_cast<std::uint64_t>(integer);
}

/**
 * @brief Adds an owned JSON string member to an object.
 *
 * @param object Destination json-c object that takes ownership of the value.
 * @param key Top-level JSON key to add.
 * @param value String bytes to serialize.
 * @return None; object owns the added json-c value after return.
 * @throws std::bad_alloc If json-c cannot allocate the string value.
 * @throws std::overflow_error If value is too large for json-c's string API.
 */
void add_string(json_object* object, const char* key, const std::string& value) {
    if (value.size() > static_cast<std::size_t>(
                           std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "state string exceeds json-c length range: " + std::string(key)
        );
    }
    json_object* encoded = json_object_new_string_len(
        value.data(),
        static_cast<int>(value.size())
    );
    if (encoded == nullptr) {
        throw std::bad_alloc();
    }
    json_object_object_add(object, key, encoded);
}

/**
 * @brief Adds a non-negative uint64 field through json-c's signed integer type.
 *
 * @param object Destination json-c object that takes ownership of the value.
 * @param key Top-level JSON key to add.
 * @param value Counter value already validated to fit in int64_t.
 * @return None; object owns the added json-c value after return.
 * @throws std::bad_alloc If json-c cannot allocate the integer value.
 */
void add_u64(json_object* object, const char* key, std::uint64_t value) {
    json_object* encoded = json_object_new_int64(
        static_cast<std::int64_t>(value)
    );
    if (encoded == nullptr) {
        throw std::bad_alloc();
    }
    json_object_object_add(object, key, encoded);
}

/**
 * @brief Serializes a validated state into its stable top-level JSON representation.
 *
 * @param state Fully validated state value to serialize.
 * @return Compact JSON text followed by one newline.
 * @throws std::invalid_argument If state is internally inconsistent.
 * @throws std::bad_alloc If json-c allocation fails.
 */
std::string serialize_state(const VideoState& state) {
    validate_state(state);
    JsonObject root(json_object_new_object());

    add_string(root.get(), "run_id", state.run_id);
    add_string(root.get(), "video_id", state.video_id);
    add_string(root.get(), "watch_url", state.watch_url);
    add_string(root.get(), "source_path", state.source_path);
    add_string(root.get(), "extractor_version", state.extractor_version);
    add_string(root.get(), "config_hash", state.config_hash);
    add_string(root.get(), "status", std::string(status_name(state.status)));
    add_string(root.get(), "started_at", state.started_at);
    add_string(root.get(), "updated_at", state.updated_at);

    if (state.last_completed_sample_index.has_value()) {
        add_u64(
            root.get(),
            "last_completed_sample_index",
            state.last_completed_sample_index.value()
        );
    } else {
        // json-c represents JSON null as a null object pointer; object_add
        // serializes that ownership-free value as the required JSON null.
        json_object_object_add(root.get(), "last_completed_sample_index", nullptr);
    }

    add_u64(root.get(), "emitted_frame_count", state.emitted_frame_count);
    add_string(root.get(), "native_manifest_path", state.native_manifest_path);
    add_string(
        root.get(),
        "enrichment_manifest_path",
        state.enrichment_manifest_path
    );
    add_string(root.get(), "error", state.error);

    const char* encoded = json_object_to_json_string_ext(
        root.get(),
        JSON_C_TO_STRING_PLAIN
    );
    if (encoded == nullptr) {
        throw std::runtime_error("unable to serialize state JSON");
    }
    return std::string(encoded) + "\n";
}

/**
 * @brief Confirms that a loaded state exactly matches the requested provenance.
 *
 * @param state Persisted state loaded from its final JSON path.
 * @param identity Run, video, version, and configuration expected by caller.
 * @return None; returns only when every immutable provenance value matches.
 * @throws std::invalid_argument If identity is invalid or any value differs.
 */
void validate_matching_identity(
    const VideoState& state,
    const StateIdentity& identity
) {
    validate_identity(identity);
    if (state.run_id != identity.run_id ||
        state.video_id != identity.video_id ||
        state.extractor_version != identity.extractor_version ||
        state.config_hash != identity.config_hash) {
        throw std::invalid_argument("state provenance does not match request");
    }
}

/**
 * @brief Determines whether one lifecycle edge is allowed by the native contract.
 *
 * @param current Persisted status before mutation.
 * @param next Requested successor status.
 * @return True when the exact transition is allowed; otherwise false.
 */
bool is_allowed_transition(VideoStatus current, VideoStatus next) {
    switch (current) {
    case VideoStatus::Pending:
        return next == VideoStatus::Downloading || next == VideoStatus::Failed;
    case VideoStatus::Downloading:
        return next == VideoStatus::Extracting || next == VideoStatus::Failed;
    case VideoStatus::Extracting:
        return next == VideoStatus::Extracted || next == VideoStatus::Failed;
    case VideoStatus::Extracted:
        return next == VideoStatus::EnrichmentPending || next == VideoStatus::Failed;
    case VideoStatus::EnrichmentPending:
        return next == VideoStatus::Enriched || next == VideoStatus::Failed;
    case VideoStatus::Enriched:
        return next == VideoStatus::Published || next == VideoStatus::Failed;
    case VideoStatus::Published:
        return next == VideoStatus::Cleaned;
    case VideoStatus::Cleaned:
        return false;
    case VideoStatus::Failed:
        return next == VideoStatus::Downloading;
    }

    return false;
}

/**
 * @brief Validates that a final state path can have a same-directory temp file.
 *
 * @param state_path Candidate final JSON destination path.
 * @return None; returns only when state_path names a file.
 * @throws std::invalid_argument If state_path is empty or has no filename.
 */
void validate_state_path(const std::filesystem::path& state_path) {
    if (state_path.empty() || state_path.filename().empty()) {
        throw std::invalid_argument("state path must name a file");
    }
}

/**
 * @brief Converts a non-empty path to a normalized absolute path.
 *
 * @param path Relative or absolute path supplied by a lifecycle command.
 * @param label Human-readable path role included in diagnostics.
 * @return Lexically normalized absolute path.
 * @throws std::invalid_argument If path is blank.
 * @throws std::system_error If the current working directory cannot be resolved.
 */
std::filesystem::path normalized_absolute_path(
    const std::filesystem::path& path,
    std::string_view label
) {
    if (path.empty()) {
        throw std::invalid_argument(std::string(label) + " must not be blank");
    }

    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(path, error);
    if (error) {
        throw std::system_error(error, "resolve " + std::string(label));
    }
    return absolute.lexically_normal();
}

/**
 * @brief Rejects a run root that could resolve to a broad filesystem target.
 *
 * @param run_root User-supplied native run root.
 * @return Normalized absolute run root with an unambiguous basename.
 * @throws std::invalid_argument If run_root is blank, root-like, or ambiguous.
 * @throws std::system_error If run_root cannot be resolved.
 */
std::filesystem::path normalized_run_root(const std::filesystem::path& run_root) {
    const std::filesystem::path normalized = normalized_absolute_path(
        run_root,
        "run_root"
    );
    if (normalized == normalized.root_path()) {
        throw std::invalid_argument("run_root must not be a filesystem root");
    }
    const std::string run_id = normalized.filename().string();
    if (run_id.empty() || run_id == "." || run_id == "..") {
        throw std::invalid_argument("run_root must have an unambiguous basename");
    }
    return normalized;
}

/**
 * @brief Tests whether a relative path attempts to leave its expected root.
 *
 * @param path Lexically relative candidate path.
 * @return True when any component is ``..``; otherwise false.
 */
bool has_parent_component(const std::filesystem::path& path) {
    for (const std::filesystem::path& component : path) {
        if (component == "..") {
            return true;
        }
    }
    return false;
}

/**
 * @brief Resolves a command path and requires it to remain below run_root.
 *
 * @param run_root Normalized absolute lifecycle root.
 * @param candidate Relative or absolute candidate file path.
 * @param label Human-readable path role included in diagnostics.
 * @return Normalized absolute candidate path confined below run_root.
 * @throws std::invalid_argument If candidate is blank, is run_root itself, or
 *         lexically escapes run_root.
 * @throws std::system_error If a path cannot be resolved.
 */
std::filesystem::path resolve_inside_run_root(
    const std::filesystem::path& run_root,
    const std::filesystem::path& candidate,
    std::string_view label
) {
    const std::filesystem::path resolved = normalized_absolute_path(
        candidate.is_absolute() ? candidate : run_root / candidate,
        label
    );
    const std::filesystem::path relative = resolved.lexically_relative(run_root);
    if (relative.empty() || relative == "." || has_parent_component(relative)) {
        throw std::invalid_argument(
            std::string(label) + " must remain below run_root"
        );
    }
    return resolved;
}

/**
 * @brief Converts a confined path to the portable run-root-relative state form.
 *
 * @param run_root Normalized absolute lifecycle root.
 * @param path Absolute path already known to be below run_root.
 * @param label Human-readable path role included in diagnostics.
 * @return Non-empty POSIX-style path relative to run_root.
 * @throws std::invalid_argument If path cannot be represented safely below root.
 */
std::string run_relative_path(
    const std::filesystem::path& run_root,
    const std::filesystem::path& path,
    std::string_view label
) {
    const std::filesystem::path resolved = resolve_inside_run_root(
        run_root,
        path,
        label
    );
    return resolved.lexically_relative(run_root).generic_string();
}

/**
 * @brief Tests whether an exact filesystem path exists without hiding errors.
 *
 * @param path Exact file or directory path to inspect.
 * @return True if path exists; false only if it is absent.
 * @throws std::system_error If inspection itself fails.
 */
bool path_exists(const std::filesystem::path& path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw std::system_error(error, "inspect path " + path.string());
    }
    return exists;
}

/**
 * @brief Requires an existing regular file for a lifecycle artifact.
 *
 * @param path Exact artifact path expected to be a non-directory file.
 * @param label Human-readable artifact role included in diagnostics.
 * @return None; returns only when path is a regular file.
 * @throws std::invalid_argument If path is absent or is not a regular file.
 * @throws std::system_error If file inspection fails.
 */
void require_regular_file(
    const std::filesystem::path& path,
    std::string_view label
) {
    std::error_code error;
    const bool regular = std::filesystem::is_regular_file(path, error);
    if (error) {
        throw std::system_error(error, "inspect " + std::string(label));
    }
    if (!regular) {
        throw std::invalid_argument(
            std::string(label) + " must be an existing regular file: " +
            path.string()
        );
    }
}

/**
 * @brief Renames one exact lifecycle artifact while preserving filesystem errors.
 *
 * @param source Existing path that transfers to destination.
 * @param destination Final path receiving source.
 * @param label Human-readable operation role included in diagnostics.
 * @return None; destination names source after a successful call.
 * @throws std::system_error If the rename cannot be completed.
 */
void rename_exact_path(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string_view label
) {
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    if (error) {
        throw std::system_error(
            error,
            std::string(label) + " " + source.string() + " to " +
                destination.string()
        );
    }
}

/**
 * @brief Removes one exact temporary file when it is present.
 *
 * @param path Exact file path derived from a validated video identifier.
 * @return None; path is absent after a successful return.
 * @throws std::system_error If removal fails.
 */
void remove_exact_file_if_present(const std::filesystem::path& path) {
    if (!path_exists(path)) {
        return;
    }

    std::error_code error;
    std::filesystem::remove(path, error);
    if (error) {
        throw std::system_error(error, "remove temporary file " + path.string());
    }
}

/**
 * @brief Removes one exact per-video directory tree when it is present.
 *
 * @param path Exact staging or rollback directory built from a safe video ID.
 * @param label Human-readable operation role included in diagnostics.
 * @return None; path is absent after a successful return.
 * @throws std::system_error If removal fails.
 */
void remove_exact_tree_if_present(
    const std::filesystem::path& path,
    std::string_view label
) {
    if (!path_exists(path)) {
        return;
    }

    std::error_code error;
    std::filesystem::remove_all(path, error);
    if (error) {
        throw std::system_error(error, std::string(label) + " " + path.string());
    }
}

/**
 * @brief Reads a required string from a non-state lifecycle JSON document.
 *
 * @param object Parsed JSON object that owns the requested member.
 * @param key Required member name.
 * @param path Source document path used in diagnostics.
 * @param document_name Human-readable document type used in diagnostics.
 * @param allow_blank Whether an empty string is accepted.
 * @return Copied string value from the JSON member.
 * @throws std::invalid_argument If the member is absent, not a string, or blank.
 */
std::string required_document_string(
    json_object* object,
    const char* key,
    const std::filesystem::path& path,
    std::string_view document_name,
    bool allow_blank = false
) {
    json_object* value = nullptr;
    if (!json_object_object_get_ex(object, key, &value) || value == nullptr ||
        !json_object_is_type(value, json_type_string)) {
        throw std::invalid_argument(
            std::string(document_name) + " must contain string '" + key +
            "': " + path.string()
        );
    }
    const char* raw = json_object_get_string(value);
    const std::string result = raw == nullptr ? std::string() : std::string(raw);
    if (!allow_blank) {
        require_non_blank(result, key);
    }
    return result;
}

/**
 * @brief Reads a required non-negative count from a lifecycle JSON document.
 *
 * @param object Parsed JSON object that owns the requested member.
 * @param key Required member name.
 * @param path Source document path used in diagnostics.
 * @param document_name Human-readable document type used in diagnostics.
 * @return Exact non-negative count represented as uint64_t.
 * @throws std::invalid_argument If the member is absent, non-integral, or negative.
 */
std::uint64_t required_document_u64(
    json_object* object,
    const char* key,
    const std::filesystem::path& path,
    std::string_view document_name
) {
    json_object* value = nullptr;
    if (!json_object_object_get_ex(object, key, &value) || value == nullptr ||
        !json_object_is_type(value, json_type_int)) {
        throw std::invalid_argument(
            std::string(document_name) + " must contain integer '" + key +
            "': " + path.string()
        );
    }
    const std::int64_t integer = json_object_get_int64(value);
    if (integer < 0) {
        throw std::invalid_argument(
            std::string(document_name) + " count must be non-negative '" + key +
            "': " + path.string()
        );
    }
    return static_cast<std::uint64_t>(integer);
}

/**
 * @brief Represents the subset of a native manifest required by state commands.
 */
struct NativeManifestSummary {
    /** @brief Canonical video identifier owned by the native bundle. */
    std::string video_id;
    /** @brief Native lifecycle token recorded in the manifest. */
    std::string status;
    /** @brief Expected deterministic one-FPS sample count. */
    std::uint64_t expected_frame_count;
    /** @brief Actual emitted native frame count. */
    std::uint64_t emitted_frame_count;
    /** @brief Native extractor provenance token. */
    std::string extractor_version;
    /** @brief Deterministic extraction configuration hash. */
    std::string config_hash;
    /** @brief Bundle-relative JSONL path containing native frame rows. */
    std::string frames_jsonl;
};

/**
 * @brief Parses the lifecycle-relevant fields of one native bundle manifest.
 *
 * @param manifest_path Existing native manifest JSON file.
 * @return Validated subset of the native manifest contract.
 * @throws std::runtime_error If manifest_path cannot be read.
 * @throws std::invalid_argument If the document is malformed or incomplete.
 */
NativeManifestSummary read_native_manifest_summary(
    const std::filesystem::path& manifest_path
) {
    require_regular_file(manifest_path, "native manifest");
    const JsonObject root = parse_object(manifest_path, read_text(manifest_path));
    return NativeManifestSummary{
        required_document_string(
            root.get(), "video_id", manifest_path, "native manifest"
        ),
        required_document_string(
            root.get(), "status", manifest_path, "native manifest"
        ),
        required_document_u64(
            root.get(), "expected_frame_count", manifest_path, "native manifest"
        ),
        required_document_u64(
            root.get(), "emitted_frame_count", manifest_path, "native manifest"
        ),
        required_document_string(
            root.get(), "extractor_version", manifest_path, "native manifest"
        ),
        required_document_string(
            root.get(), "config_hash", manifest_path, "native manifest"
        ),
        required_document_string(
            root.get(), "frames_jsonl", manifest_path, "native manifest"
        ),
    };
}

/**
 * @brief Validates a native manifest against the state that owns its bundle.
 *
 * @param manifest_path Existing native manifest path below run_root.
 * @param state Persisted video state expected to own the manifest.
 * @param expected_status Required native manifest lifecycle token.
 * @return Parsed native manifest subset after all state checks pass.
 * @throws std::invalid_argument If identity, counts, status, or JSONL coverage fail.
 */
NativeManifestSummary validate_native_manifest_for_state(
    const std::filesystem::path& manifest_path,
    const VideoState& state,
    std::string_view expected_status
) {
    const NativeManifestSummary manifest = read_native_manifest_summary(
        manifest_path
    );
    if (manifest.video_id != state.video_id ||
        manifest.extractor_version != state.extractor_version ||
        manifest.config_hash != state.config_hash ||
        manifest.status != expected_status) {
        throw std::invalid_argument("native manifest provenance or status mismatch");
    }
    if (manifest.expected_frame_count != manifest.emitted_frame_count ||
        manifest.emitted_frame_count != state.emitted_frame_count) {
        throw std::invalid_argument("native manifest frame count mismatch");
    }

    const std::filesystem::path frames_path = (
        manifest_path.parent_path() / std::filesystem::path(manifest.frames_jsonl)
    ).lexically_normal();
    const std::filesystem::path relative = frames_path.lexically_relative(
        manifest_path.parent_path()
    );
    if (relative.empty() || relative == "." || has_parent_component(relative)) {
        throw std::invalid_argument("native manifest frames_jsonl escapes its bundle");
    }
    require_regular_file(frames_path, "native frame JSONL");
    return manifest;
}

/**
 * @brief Represents the compact provenance accepted after specialist enrichment.
 */
struct EnrichmentHandoff {
    /** @brief Canonical video identifier represented by every artifact. */
    std::string video_id;
    /** @brief Exact frame count validated by the Python handoff writer. */
    std::uint64_t frame_count;
    /** @brief Native manifest path recorded by the handoff writer. */
    std::string native_manifest_path;
    /** @brief SHA-256 digest of ordered native frame IDs. */
    std::string frame_id_digest;
    /** @brief Isolated custom frame store identifier. */
    std::string frame_store_id;
    /** @brief Extraction configuration hash shared with native state. */
    std::string config_hash;
};

/**
 * @brief Parses and requires all specialist artifact entries in an enrichment handoff.
 *
 * @param handoff_path Existing compact handoff JSON path.
 * @return Validated handoff provenance without loading the specialist artifacts.
 * @throws std::runtime_error If handoff_path cannot be read.
 * @throws std::invalid_argument If required fields or artifact entries are missing.
 */
EnrichmentHandoff read_enrichment_handoff(
    const std::filesystem::path& handoff_path
) {
    require_regular_file(handoff_path, "enrichment handoff");
    const JsonObject root = parse_object(handoff_path, read_text(handoff_path));
    json_object* artifacts = nullptr;
    if (!json_object_object_get_ex(root.get(), "artifacts", &artifacts) ||
        artifacts == nullptr || !json_object_is_type(artifacts, json_type_object)) {
        throw std::invalid_argument(
            "enrichment handoff must contain an artifacts object: " +
            handoff_path.string()
        );
    }

    for (const char* key : {"caption", "ocr", "objects", "asr"}) {
        json_object* artifact = nullptr;
        if (!json_object_object_get_ex(artifacts, key, &artifact) ||
            artifact == nullptr || !json_object_is_type(artifact, json_type_object)) {
            throw std::invalid_argument(
                "enrichment handoff is missing artifact entry '" +
                std::string(key) + "': " + handoff_path.string()
            );
        }
        const std::string status = required_document_string(
            artifact,
            "status",
            handoff_path,
            "enrichment artifact"
        );
        const std::string artifact_path = required_document_string(
            artifact,
            "path",
            handoff_path,
            "enrichment artifact",
            status == "not_evaluated"
        );
        if (status != "not_evaluated") {
            require_non_blank(artifact_path, "artifact path");
        }
    }

    return EnrichmentHandoff{
        required_document_string(
            root.get(), "video_id", handoff_path, "enrichment handoff"
        ),
        required_document_u64(
            root.get(), "frame_count", handoff_path, "enrichment handoff"
        ),
        required_document_string(
            root.get(), "native_manifest_path", handoff_path, "enrichment handoff"
        ),
        required_document_string(
            root.get(), "frame_id_digest", handoff_path, "enrichment handoff"
        ),
        required_document_string(
            root.get(), "frame_store_id", handoff_path, "enrichment handoff"
        ),
        required_document_string(
            root.get(), "config_hash", handoff_path, "enrichment handoff"
        ),
    };
}

/**
 * @brief Validates enrichment provenance against the current native bundle state.
 *
 * @param run_root Normalized absolute lifecycle root.
 * @param handoff_path Existing handoff path below run_root.
 * @param native_manifest_path Existing native manifest path below run_root.
 * @param state State expected to own both documents.
 * @return None; returns only when all handoff and native values agree.
 * @throws std::invalid_argument If identity, configuration, paths, or counts differ.
 */
void validate_handoff_against_state(
    const std::filesystem::path& run_root,
    const std::filesystem::path& handoff_path,
    const std::filesystem::path& native_manifest_path,
    const VideoState& state
) {
    const EnrichmentHandoff handoff = read_enrichment_handoff(handoff_path);
    const NativeManifestSummary native_manifest = validate_native_manifest_for_state(
        native_manifest_path,
        state,
        "enrichment_pending"
    );
    const std::filesystem::path declared_native_manifest = resolve_inside_run_root(
        run_root,
        handoff.native_manifest_path,
        "handoff native_manifest_path"
    );
    if (declared_native_manifest != native_manifest_path ||
        handoff.video_id != state.video_id || handoff.config_hash != state.config_hash ||
        handoff.frame_count != native_manifest.emitted_frame_count) {
        throw std::invalid_argument("enrichment handoff provenance mismatch");
    }
}

/**
 * @brief Serializes an existing native manifest with a final published status.
 *
 * @param manifest_path Existing enrichment-pending native manifest.
 * @return Compact published manifest JSON followed by one newline.
 * @throws std::runtime_error If manifest_path cannot be read or JSON cannot serialize.
 * @throws std::invalid_argument If the JSON root is invalid.
 */
std::string serialize_published_manifest(
    const std::filesystem::path& manifest_path
) {
    const JsonObject root = parse_object(manifest_path, read_text(manifest_path));
    json_object* status = json_object_new_string("published");
    if (status == nullptr) {
        throw std::bad_alloc();
    }
    json_object_object_add(root.get(), "status", status);
    const char* encoded = json_object_to_json_string_ext(
        root.get(),
        JSON_C_TO_STRING_PLAIN
    );
    if (encoded == nullptr) {
        throw std::runtime_error("unable to serialize published native manifest");
    }
    return std::string(encoded) + "\n";
}

/**
 * @brief Writes text through a same-directory temporary artifact and rename.
 *
 * @param destination Final file path that receives content.
 * @param content Complete byte sequence to publish.
 * @param label Human-readable artifact role used in diagnostics.
 * @return None; readers observe an old complete file or the new complete file.
 * @throws std::invalid_argument If destination does not name a file.
 * @throws std::runtime_error If temporary output cannot be written.
 * @throws std::system_error If parent creation or atomic rename fails.
 */
void write_text_atomic(
    const std::filesystem::path& destination,
    const std::string& content,
    std::string_view label
) {
    if (destination.empty() || destination.filename().empty()) {
        throw std::invalid_argument(std::string(label) + " must name a file");
    }
    std::error_code error;
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error) {
        throw std::system_error(
            error,
            "create " + std::string(label) + " directory"
        );
    }

    std::filesystem::path temporary = destination;
    temporary += ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "unable to open temporary " + std::string(label) + ": " +
                temporary.string()
            );
        }
        output << content;
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "unable to flush temporary " + std::string(label) + ": " +
                temporary.string()
            );
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "unable to close temporary " + std::string(label) + ": " +
                temporary.string()
            );
        }
        rename_exact_path(temporary, destination, "publish");
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw;
    }
}

/**
 * @brief Reconstructs immutable state identity directly from a loaded checkpoint.
 *
 * @param state Persisted state whose immutable provenance is copied.
 * @return StateIdentity with the exact values recorded by extraction.
 */
StateIdentity identity_from_state(const VideoState& state) {
    return StateIdentity{
        state.run_id,
        state.video_id,
        state.extractor_version,
        state.config_hash,
    };
}

/**
 * @brief Builds the exact state file path for one validated video identifier.
 *
 * @param run_root Normalized absolute lifecycle root.
 * @param video_id Canonical source video identifier.
 * @return Absolute ``state/{video_id}.json`` path below run_root.
 * @throws std::invalid_argument If video_id is not a safe native identifier.
 */
std::filesystem::path state_path_for(
    const std::filesystem::path& run_root,
    const std::string& video_id
) {
    static_cast<void>(make_frame_id(video_id, 0));
    return run_root / "state" / (video_id + ".json");
}

/**
 * @brief Reads a state file and verifies it belongs to the selected run/video.
 *
 * @param run_root Normalized absolute lifecycle root.
 * @param video_id Canonical identifier requested by the CLI.
 * @return Fully validated state owned by the requested root and video.
 * @throws std::invalid_argument If the state was moved from another run or video.
 * @throws std::runtime_error If the state cannot be read.
 */
VideoState read_owned_state(
    const std::filesystem::path& run_root,
    const std::string& video_id
) {
    const VideoState state = read_state(state_path_for(run_root, video_id));
    if (state.video_id != video_id || state.run_id != run_root.filename().string()) {
        throw std::invalid_argument("state does not belong to requested run/video");
    }
    return state;
}

/**
 * @brief Persists a guarded forward lifecycle transition with atomic field updates.
 *
 * @param state_path Final state JSON file to replace.
 * @param identity Immutable expected state provenance.
 * @param expected_status Required predecessor lifecycle status.
 * @param next_status Allowed successor lifecycle status.
 * @param update Callback that updates mutable provenance before status changes.
 * @return State after the single atomic transition write.
 * @throws std::invalid_argument If state provenance, predecessor, or edge is invalid.
 * @throws std::runtime_error If state cannot be read or written.
 */
VideoState transition_state_with_update(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    VideoStatus expected_status,
    VideoStatus next_status,
    const std::function<void(VideoState&)>& update
) {
    VideoState state = read_state(state_path);
    validate_matching_identity(state, identity);
    if (state.status != expected_status) {
        throw std::invalid_argument(
            "state predecessor mismatch: expected " +
            std::string(status_name(expected_status)) + ", found " +
            std::string(status_name(state.status))
        );
    }
    if (!is_allowed_transition(state.status, next_status)) {
        throw std::invalid_argument(
            "invalid state transition from " +
            std::string(status_name(state.status)) + " to " +
            std::string(status_name(next_status))
        );
    }

    update(state);
    state.status = next_status;
    state.error.clear();
    state.updated_at = utc_timestamp();
    save_state_atomic(state_path, state);
    return state;
}

}  // namespace

/**
 * @brief Creates the first durable pending checkpoint for one source video.
 *
 * @param identity Immutable run, video, extractor-version, and config identity.
 * @param watch_url Literal source watch URL retained for source acquisition.
 * @return Valid pending VideoState with equal creation and update timestamps.
 * @throws std::invalid_argument If identity or watch_url is blank.
 */
VideoState make_pending_state(
    const StateIdentity& identity,
    std::string watch_url
) {
    validate_identity(identity);
    require_non_blank(watch_url, "watch_url");

    const std::string timestamp = utc_timestamp();
    return VideoState{
        identity.run_id,
        identity.video_id,
        std::move(watch_url),
        "",
        identity.extractor_version,
        identity.config_hash,
        VideoStatus::Pending,
        timestamp,
        timestamp,
        std::nullopt,
        0,
        "",
        "",
        "",
    };
}

/**
 * @brief Reads one JSON checkpoint and enforces its native state invariants.
 *
 * @param state_path Final state JSON path to parse.
 * @return Fully validated state reconstructed from the top-level JSON fields.
 * @throws std::runtime_error If state_path cannot be opened or read.
 * @throws std::invalid_argument If JSON shape or values violate the contract.
 */
VideoState read_state(const std::filesystem::path& state_path) {
    const JsonObject root = parse_object(state_path, read_text(state_path));

    VideoState state{
        required_string(root.get(), "run_id", state_path),
        required_string(root.get(), "video_id", state_path),
        required_string(root.get(), "watch_url", state_path),
        required_string(root.get(), "source_path", state_path),
        required_string(root.get(), "extractor_version", state_path),
        required_string(root.get(), "config_hash", state_path),
        parse_status(required_string(root.get(), "status", state_path)),
        required_string(root.get(), "started_at", state_path),
        required_string(root.get(), "updated_at", state_path),
        required_optional_u64(
            root.get(),
            "last_completed_sample_index",
            state_path
        ),
        required_u64(root.get(), "emitted_frame_count", state_path),
        required_string(root.get(), "native_manifest_path", state_path),
        required_string(root.get(), "enrichment_manifest_path", state_path),
        required_string(root.get(), "error", state_path),
    };
    validate_state(state);
    return state;
}

/**
 * @brief Atomically writes one fully validated checkpoint JSON document.
 *
 * @param state_path Final state JSON path to replace through `<state_path>.tmp`.
 * @param state Valid checkpoint data to serialize and publish.
 * @return None; readers see a complete old or complete new state document.
 * @throws std::invalid_argument If state_path or state is invalid.
 * @throws std::runtime_error If the temporary file cannot be written and closed.
 * @throws std::system_error If directory creation or final rename fails.
 */
void save_state_atomic(
    const std::filesystem::path& state_path,
    const VideoState& state
) {
    validate_state_path(state_path);
    const std::string serialized = serialize_state(state);

    const std::filesystem::path parent = state_path.parent_path();
    std::error_code filesystem_error;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, filesystem_error);
        if (filesystem_error) {
            throw std::system_error(
                filesystem_error,
                "create state directory " + parent.string()
            );
        }
    }

    std::filesystem::path temporary_path = state_path;
    temporary_path += ".tmp";
    try {
        std::ofstream output(
            temporary_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open temporary state file: " +
                temporary_path.string()
            );
        }

        output << serialized;
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "unable to flush temporary state file: " +
                temporary_path.string()
            );
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "unable to close temporary state file: " +
                temporary_path.string()
            );
        }

        std::filesystem::rename(temporary_path, state_path, filesystem_error);
        if (filesystem_error) {
            throw std::system_error(
                filesystem_error,
                "atomically replace state file " + state_path.string()
            );
        }
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw;
    }
}

/**
 * @brief Guards provenance and persists one allowed native lifecycle transition.
 *
 * @param state_path Final JSON state path to replace atomically.
 * @param identity Immutable provenance expected by the current run.
 * @param expected_status Required persisted predecessor state.
 * @param next_status Requested successor state.
 * @param error Bounded failure diagnostic required for a Failed successor.
 * @return Persisted state after a successful transition.
 * @throws std::invalid_argument If provenance, predecessor, edge, or error is invalid.
 * @throws std::runtime_error If the checkpoint cannot be read or written.
 * @throws std::system_error If atomic replacement fails.
 */
VideoState transition_state(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    VideoStatus expected_status,
    VideoStatus next_status,
    std::string error
) {
    VideoState state = read_state(state_path);
    validate_matching_identity(state, identity);
    if (state.status != expected_status) {
        throw std::invalid_argument(
            "state predecessor mismatch: expected " +
            std::string(status_name(expected_status)) + ", found " +
            std::string(status_name(state.status))
        );
    }
    if (!is_allowed_transition(state.status, next_status)) {
        throw std::invalid_argument(
            "invalid state transition from " +
            std::string(status_name(state.status)) + " to " +
            std::string(status_name(next_status))
        );
    }

    if (next_status == VideoStatus::Failed) {
        state.error = bounded_error(std::move(error));
        require_non_blank(state.error, "error");
    } else {
        if (!error.empty()) {
            throw std::invalid_argument(
                "only failed transitions may provide an error diagnostic"
            );
        }
        state.error.clear();
    }

    state.status = next_status;
    state.updated_at = utc_timestamp();
    save_state_atomic(state_path, state);
    return state;
}

/**
 * @brief Accepts a compact specialist-enrichment handoff for one native video.
 *
 * @param run_root Root containing the selected video's state and staging bundle.
 * @param video_id Canonical source-video identifier selected by the caller.
 * @param handoff_path Existing compact handoff JSON produced by Python.
 * @return None; accepts the handoff once or performs an exact idempotent no-op.
 * @throws std::invalid_argument If state, native manifest, or handoff provenance
 *         does not agree with the selected lifecycle.
 * @throws std::runtime_error If a required state or artifact cannot be read.
 * @throws std::system_error If the state replacement cannot be published.
 */
void mark_video_enriched(
    const std::filesystem::path& run_root,
    const std::string& video_id,
    const std::filesystem::path& handoff_path
) {
    const std::filesystem::path root = normalized_run_root(run_root);
    const std::filesystem::path state_path = state_path_for(root, video_id);
    const VideoState state = read_owned_state(root, video_id);
    const std::filesystem::path native_manifest_path = resolve_inside_run_root(
        root,
        state.native_manifest_path,
        "state native_manifest_path"
    );
    const std::filesystem::path resolved_handoff_path = resolve_inside_run_root(
        root,
        handoff_path,
        "handoff path"
    );
    validate_handoff_against_state(
        root,
        resolved_handoff_path,
        native_manifest_path,
        state
    );
    const std::string handoff_relative = run_relative_path(
        root,
        resolved_handoff_path,
        "handoff path"
    );

    if (state.status == VideoStatus::Enriched) {
        if (state.enrichment_manifest_path != handoff_relative) {
            throw std::invalid_argument(
                "a different enrichment handoff cannot replace accepted provenance"
            );
        }
        return;
    }
    if (state.status != VideoStatus::EnrichmentPending) {
        throw std::invalid_argument(
            "mark-enriched requires enrichment_pending state"
        );
    }

    transition_state_with_update(
        state_path,
        identity_from_state(state),
        VideoStatus::EnrichmentPending,
        VideoStatus::Enriched,
        [&handoff_relative](VideoState& updated) {
            updated.enrichment_manifest_path = handoff_relative;
        }
    );
}

/**
 * @brief Restores staging and a prior published bundle after a failed publication.
 *
 * @param staging_bundle Original extraction bundle path.
 * @param published_bundle Final published bundle path.
 * @param candidate_bundle Hidden in-progress bundle path.
 * @param backup_bundle Hidden preserved predecessor bundle path.
 * @param original_manifest Complete pre-publication native manifest content.
 * @param staging_moved Whether staging was renamed to the candidate directory.
 * @param candidate_published Whether the candidate was renamed to published.
 * @param predecessor_moved Whether an old published bundle was moved to backup.
 * @return None; restores the old observable bundle layout when paths are present.
 * @throws std::system_error If an exact rollback rename cannot be completed.
 */
void rollback_failed_publication(
    const std::filesystem::path& staging_bundle,
    const std::filesystem::path& published_bundle,
    const std::filesystem::path& candidate_bundle,
    const std::filesystem::path& backup_bundle,
    const std::string& original_manifest,
    bool staging_moved,
    bool candidate_published,
    bool predecessor_moved
) {
    if (candidate_published && path_exists(published_bundle)) {
        rename_exact_path(
            published_bundle,
            candidate_bundle,
            "rollback published candidate"
        );
    }
    if (staging_moved && path_exists(candidate_bundle)) {
        write_text_atomic(
            candidate_bundle / "manifest.json",
            original_manifest,
            "rollback native manifest"
        );
        rename_exact_path(
            candidate_bundle,
            staging_bundle,
            "restore staging bundle"
        );
    }
    if (predecessor_moved && path_exists(backup_bundle)) {
        rename_exact_path(
            backup_bundle,
            published_bundle,
            "restore previous published bundle"
        );
    }
}

/**
 * @brief Moves an enriched staging bundle into its durable published location.
 *
 * @param run_root Root containing the selected video's state and lifecycle data.
 * @param video_id Canonical source-video identifier selected by the caller.
 * @param manifest_path Exact staging native manifest provided as a publication guard.
 * @return None; publishes the final manifest before transitioning state.
 * @throws std::invalid_argument If source state, handoff, or native provenance differs.
 * @throws std::runtime_error If publication artifacts cannot be read or restored.
 * @throws std::system_error If a filesystem publication operation fails.
 */
void mark_video_published(
    const std::filesystem::path& run_root,
    const std::string& video_id,
    const std::filesystem::path& manifest_path
) {
    const std::filesystem::path root = normalized_run_root(run_root);
    const std::filesystem::path state_path = state_path_for(root, video_id);
    const VideoState state = read_owned_state(root, video_id);
    const std::filesystem::path requested_manifest_path = resolve_inside_run_root(
        root,
        manifest_path,
        "publication manifest path"
    );
    const std::filesystem::path state_native_manifest_path =
        resolve_inside_run_root(
            root,
            state.native_manifest_path,
            "state native_manifest_path"
        );

    if (state.status == VideoStatus::Published) {
        if (requested_manifest_path != state_native_manifest_path) {
            throw std::invalid_argument(
                "mark-published idempotency requires the accepted published manifest"
            );
        }
        static_cast<void>(validate_native_manifest_for_state(
            state_native_manifest_path,
            state,
            "published"
        ));
        return;
    }
    if (state.status != VideoStatus::Enriched) {
        throw std::invalid_argument("mark-published requires enriched state");
    }
    if (requested_manifest_path != state_native_manifest_path) {
        throw std::invalid_argument(
            "publication manifest must match the state native manifest"
        );
    }
    if (state.enrichment_manifest_path.empty()) {
        throw std::invalid_argument("enriched state must retain a handoff path");
    }

    const std::filesystem::path handoff_path = resolve_inside_run_root(
        root,
        state.enrichment_manifest_path,
        "state enrichment_manifest_path"
    );
    validate_handoff_against_state(
        root,
        handoff_path,
        state_native_manifest_path,
        state
    );
    static_cast<void>(validate_native_manifest_for_state(
        state_native_manifest_path,
        state,
        "enrichment_pending"
    ));

    const std::filesystem::path staging_bundle = root / "staging" / video_id;
    const std::filesystem::path expected_staging_manifest =
        staging_bundle / "manifest.json";
    if (state_native_manifest_path != expected_staging_manifest) {
        throw std::invalid_argument(
            "enriched native manifest must remain in the selected staging bundle"
        );
    }
    const std::filesystem::path handoff_relative = handoff_path.lexically_relative(
        staging_bundle
    );
    if (handoff_relative.empty() || handoff_relative == "." ||
        has_parent_component(handoff_relative)) {
        throw std::invalid_argument(
            "enrichment handoff must remain within the selected staging bundle"
        );
    }

    const std::string original_manifest = read_text(state_native_manifest_path);
    const std::string published_manifest = serialize_published_manifest(
        state_native_manifest_path
    );
    const std::filesystem::path published_root = root / "published";
    const std::filesystem::path published_bundle = published_root / video_id;
    const std::filesystem::path candidate_bundle = published_root /
        ("." + video_id + ".publishing");
    const std::filesystem::path backup_bundle = published_root /
        ("." + video_id + ".previous");
    if (path_exists(candidate_bundle) || path_exists(backup_bundle)) {
        throw std::runtime_error(
            "refusing publication with unresolved per-video rollback artifacts"
        );
    }

    std::error_code directory_error;
    std::filesystem::create_directories(published_root, directory_error);
    if (directory_error) {
        throw std::system_error(directory_error, "create published root");
    }

    const bool had_previous_published = path_exists(published_bundle);
    bool predecessor_moved = false;
    bool staging_moved = false;
    bool candidate_published = false;
    try {
        if (had_previous_published) {
            rename_exact_path(
                published_bundle,
                backup_bundle,
                "preserve previous published bundle"
            );
            predecessor_moved = true;
        }
        rename_exact_path(
            staging_bundle,
            candidate_bundle,
            "move staging bundle into publication candidate"
        );
        staging_moved = true;
        remove_exact_file_if_present(candidate_bundle / "manifest.json");
        rename_exact_path(
            candidate_bundle,
            published_bundle,
            "publish complete native bundle"
        );
        candidate_published = true;
        write_text_atomic(
            published_bundle / "manifest.json",
            published_manifest,
            "published native manifest"
        );

        const std::string published_native_relative = run_relative_path(
            root,
            published_bundle / "manifest.json",
            "published native manifest"
        );
        const std::string published_handoff_relative = run_relative_path(
            root,
            published_bundle / handoff_relative,
            "published enrichment handoff"
        );
        transition_state_with_update(
            state_path,
            identity_from_state(state),
            VideoStatus::Enriched,
            VideoStatus::Published,
            [&published_native_relative, &published_handoff_relative](
                VideoState& updated
            ) {
                updated.native_manifest_path = published_native_relative;
                updated.enrichment_manifest_path = published_handoff_relative;
            }
        );

        if (predecessor_moved) {
            std::error_code cleanup_error;
            std::filesystem::remove_all(backup_bundle, cleanup_error);
        }
    } catch (...) {
        rollback_failed_publication(
            staging_bundle,
            published_bundle,
            candidate_bundle,
            backup_bundle,
            original_manifest,
            staging_moved,
            candidate_published,
            predecessor_moved
        );
        throw;
    }
}

/**
 * @brief Removes temporary artifacts only after a complete published bundle exists.
 *
 * @param run_root Root containing the selected video's native lifecycle data.
 * @param video_id Canonical source-video identifier selected by the caller.
 * @return None; deletes only known source/staging and published OCR scratch
 *         paths, then checkpoints without removing durable published evidence.
 * @throws std::invalid_argument If state predecessor or published provenance is invalid.
 * @throws std::runtime_error If required state/manifest files cannot be read.
 * @throws std::system_error If scoped temporary cleanup fails.
 */
void cleanup_video(
    const std::filesystem::path& run_root,
    const std::string& video_id
) {
    const std::filesystem::path root = normalized_run_root(run_root);
    const std::filesystem::path state_path = state_path_for(root, video_id);
    const VideoState state = read_owned_state(root, video_id);
    if (state.status == VideoStatus::Cleaned) {
        return;
    }
    if (state.status != VideoStatus::Published) {
        throw std::invalid_argument("cleanup requires published state");
    }

    const std::filesystem::path published_manifest_path = resolve_inside_run_root(
        root,
        state.native_manifest_path,
        "state native_manifest_path"
    );
    const std::filesystem::path expected_published_manifest =
        root / "published" / video_id / "manifest.json";
    if (published_manifest_path != expected_published_manifest) {
        throw std::invalid_argument(
            "published state must retain the selected published native manifest"
        );
    }
    static_cast<void>(validate_native_manifest_for_state(
        published_manifest_path,
        state,
        "published"
    ));

    const std::filesystem::path source_root = root / "source";
    remove_exact_file_if_present(source_root / (video_id + ".part"));
    remove_exact_file_if_present(source_root / (video_id + ".part.part"));
    remove_exact_file_if_present(source_root / (video_id + ".part.tmp"));
    remove_exact_file_if_present(source_root / (video_id + ".part.ytdl"));
    remove_exact_tree_if_present(
        root / "staging" / video_id,
        "remove selected staging bundle"
    );
    // Publication transfers the complete staging bundle. Remove the selected
    // high-resolution OCR scratch directory only after the validated handoff
    // and publication marker exist, while preserving durable images and all
    // specialist artifacts for the published video.
    remove_exact_tree_if_present(
        root / "published" / video_id / "enrichment_images",
        "remove selected published OCR scratch images"
    );

    transition_state_with_update(
        state_path,
        identity_from_state(state),
        VideoStatus::Published,
        VideoStatus::Cleaned,
        [](VideoState&) {}
    );
}

}  // namespace hcmai::keyframes_extraction
