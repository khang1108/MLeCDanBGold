/**
 * @file state.cpp
 * @brief Implements atomic JSON video checkpoints and guarded lifecycle changes.
 *
 * This module owns native state serialization, provenance validation, and
 * allowed status transitions. It does not download sources, inspect bundles,
 * or remove staging data; later lifecycle commands compose these primitives.
 */

#include "hcmai/keyframes_extraction/state.hpp"

#include <json-c/json.h>
#include <json-c/json_tokener.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
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

}  // namespace hcmai::keyframes_extraction
