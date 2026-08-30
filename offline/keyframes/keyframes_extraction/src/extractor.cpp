/**
 * @file extractor.cpp
 * @brief Implements bounded native source preparation and one-video extraction.
 *
 * This module owns the offline lifecycle from a VideoInput row through a
 * validated staging bundle and enrichment_pending state. It intentionally does
 * not call enrichment models, publish durable bundles, or materialize Parquet.
 */

#include "hcmai/keyframes_extraction/extractor.hpp"

#include "hcmai/keyframes_extraction/config.hpp"
#include "hcmai/keyframes_extraction/disk_guard.hpp"
#include "hcmai/keyframes_extraction/ffmpeg.hpp"
#include "hcmai/keyframes_extraction/frame_index.hpp"
#include "hcmai/keyframes_extraction/jsonl.hpp"
#include "hcmai/keyframes_extraction/process.hpp"
#include "hcmai/keyframes_extraction/state.hpp"
#include "hcmai/keyframes_extraction/timestamp_sampler.hpp"

#include <json-c/json.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace hcmai::keyframes_extraction {
namespace {

/**
 * @brief Owns one json-c object and releases its reference on scope exit.
 *
 * This local wrapper is used only for native manifest serialization. It keeps
 * json-c ownership explicit without extending JSON behavior beyond this module.
 */
class JsonObject {
public:
    /**
     * @brief Takes ownership of one json-c object reference.
     *
     * @param value Owned json-c object pointer, which may be null before checks.
     * @return None; constructors do not return values.
     */
    explicit JsonObject(json_object* value) : value_(value) {}

    /**
     * @brief Releases the owned json-c reference when one is present.
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
     * @brief Provides borrowed access to the owned json-c object.
     *
     * @return Raw json-c object pointer; ownership remains with this wrapper.
     */
    json_object* get() const noexcept {
        return value_;
    }

private:
    /** @brief Owned json-c reference released by the destructor. */
    json_object* value_;
};

/**
 * @brief Adds one allocated json-c value to an object while transferring ownership.
 *
 * @param object Destination json-c object that receives the named member.
 * @param key Literal non-empty member name.
 * @param value Fresh json-c value whose ownership transfers on success.
 * @return None; object owns value after a successful call.
 * @throws std::runtime_error If object/value allocation or insertion fails.
 */
void add_json_value(
    json_object* object,
    const char* key,
    json_object* value
) {
    if (object == nullptr || key == nullptr || value == nullptr) {
        if (value != nullptr) {
            json_object_put(value);
        }
        throw std::runtime_error("unable to allocate native manifest JSON value");
    }

    if (json_object_object_add(object, key, value) != 0) {
        json_object_put(value);
        throw std::runtime_error(
            "unable to add native manifest JSON member: " + std::string(key)
        );
    }
}

/**
 * @brief Adds a string member to a native manifest JSON object.
 *
 * @param object Destination json-c object.
 * @param key Literal manifest member name.
 * @param value String value to serialize through json-c escaping.
 * @return None; appends the named member to object.
 * @throws std::runtime_error If json-c cannot create or add the member.
 */
void add_json_string(
    json_object* object,
    const char* key,
    const std::string& value
) {
    add_json_value(object, key, json_object_new_string(value.c_str()));
}

/**
 * @brief Adds a signed integer member to a native manifest JSON object.
 *
 * @param object Destination json-c object.
 * @param key Literal manifest member name.
 * @param value Signed integer value to serialize.
 * @return None; appends the named member to object.
 * @throws std::runtime_error If json-c cannot create or add the member.
 */
void add_json_integer(
    json_object* object,
    const char* key,
    std::int64_t value
) {
    add_json_value(object, key, json_object_new_int64(value));
}

/**
 * @brief Adds a bounded unsigned count to a native manifest JSON object.
 *
 * json-c's integer constructor is signed, so this helper rejects values that
 * cannot retain their exact non-negative value in the serialized manifest.
 *
 * @param object Destination json-c object.
 * @param key Literal manifest member name.
 * @param value Unsigned counter to serialize.
 * @return None; appends the named member to object.
 * @throws std::overflow_error If value exceeds the signed JSON integer range.
 * @throws std::runtime_error If json-c cannot create or add the member.
 */
void add_json_unsigned(
    json_object* object,
    const char* key,
    std::uint64_t value
) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "native manifest count exceeds JSON integer range: " +
            std::string(key)
        );
    }

    add_json_integer(object, key, static_cast<std::int64_t>(value));
}

/**
 * @brief Adds a finite floating-point member to a native manifest JSON object.
 *
 * @param object Destination json-c object.
 * @param key Literal manifest member name.
 * @param value Finite numeric value to serialize.
 * @return None; appends the named member to object.
 * @throws std::invalid_argument If value is non-finite.
 * @throws std::runtime_error If json-c cannot create or add the member.
 */
void add_json_double(
    json_object* object,
    const char* key,
    double value
) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "native manifest floating-point value must be finite: " +
            std::string(key)
        );
    }

    add_json_value(object, key, json_object_new_double(value));
}

/**
 * @brief Checks whether a path currently exists without hiding filesystem errors.
 *
 * @param path Candidate filesystem path.
 * @return True when path exists; false only when it is absent.
 * @throws std::system_error If the existence check itself fails.
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
 * @brief Creates a directory hierarchy or reports the exact filesystem failure.
 *
 * @param directory Directory path to create when absent.
 * @return None; directory exists after a successful return.
 * @throws std::invalid_argument If directory is blank.
 * @throws std::system_error If the hierarchy cannot be created.
 */
void ensure_directory(const std::filesystem::path& directory) {
    if (directory.empty()) {
        throw std::invalid_argument("directory path must not be blank");
    }

    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::system_error(error, "create directory " + directory.string());
    }
}

/**
 * @brief Removes one exact file-system entry when it is present.
 *
 * This helper never traverses a directory tree. It is used for same-directory
 * temporary files and exact source artifacts whose names are derived from a
 * validated video ID.
 *
 * @param path Exact file or empty directory path to remove.
 * @return None; path is absent after a successful return.
 * @throws std::system_error If removal fails.
 */
void remove_exact_path(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::remove(path, error);
    if (error) {
        throw std::system_error(error, "remove path " + path.string());
    }
}

/**
 * @brief Removes one validated per-video staging directory tree.
 *
 * @param directory Exact `staging/{video_id}` path constructed from a safe ID.
 * @return None; directory is absent after a successful return.
 * @throws std::system_error If the staging tree cannot be removed.
 */
void remove_staging_directory(const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    if (error) {
        throw std::system_error(
            error,
            "remove stale per-video staging directory " + directory.string()
        );
    }
}

/**
 * @brief Renames a same-filesystem temporary artifact onto its final path.
 *
 * @param temporary_path Complete temporary file path in the final path's directory.
 * @param final_path Destination path that receives the completed artifact.
 * @return None; final_path names the published artifact after success.
 * @throws std::system_error If the atomic same-filesystem rename fails.
 */
void publish_by_rename(
    const std::filesystem::path& temporary_path,
    const std::filesystem::path& final_path
) {
    std::error_code error;
    std::filesystem::rename(temporary_path, final_path, error);
    if (error) {
        throw std::system_error(
            error,
            "publish " + temporary_path.string() + " to " + final_path.string()
        );
    }
}

/**
 * @brief Validates request paths and derives a stable run identifier.
 *
 * The basename of run_root is sufficient because state files live beneath that
 * root. Rejecting `.` and root paths avoids an ambiguous run identifier and
 * prevents retry cleanup from operating under a broad filesystem root.
 *
 * @param run_root Native run root supplied by the caller.
 * @return Non-blank basename used as the immutable StateIdentity run ID.
 * @throws std::invalid_argument If run_root is blank, root-like, or ambiguous.
 */
std::string derive_run_id(const std::filesystem::path& run_root) {
    if (run_root.empty()) {
        throw std::invalid_argument("run_root must not be blank");
    }

    const std::filesystem::path normalized = run_root.lexically_normal();
    if (normalized == normalized.root_path()) {
        throw std::invalid_argument("run_root must not be a filesystem root");
    }

    const std::string run_id = normalized.filename().string();
    if (run_id.empty() || run_id == "." || run_id == "..") {
        throw std::invalid_argument("run_root must have an unambiguous basename");
    }
    return run_id;
}

/**
 * @brief Ensures an ID can safely form all per-video native file names.
 *
 * @param video_id Canonical manifest video identifier.
 * @return None; returns only when make_frame_id accepts video_id.
 * @throws std::invalid_argument If video_id is blank or contains unsafe bytes.
 */
void validate_video_id(const std::string& video_id) {
    static_cast<void>(make_frame_id(video_id, 0));
}

/**
 * @brief Builds a relative source path recorded in the durable state JSON.
 *
 * @param video_id Validated canonical video identifier.
 * @return POSIX-style run-root-relative source path for video_id.
 */
std::string source_relative_path(const std::string& video_id) {
    return (std::filesystem::path("source") / (video_id + ".part"))
        .generic_string();
}

/**
 * @brief Builds a relative per-video native manifest path for state JSON.
 *
 * @param video_id Validated canonical video identifier.
 * @return POSIX-style run-root-relative native manifest path for video_id.
 */
std::string native_manifest_relative_path(const std::string& video_id) {
    return (
        std::filesystem::path("staging") / video_id / "manifest.json"
    ).generic_string();
}

/**
 * @brief Builds a zero-padded JPEG filename from a native sample index.
 *
 * @param sample_index Zero-based custom sampling ordinal.
 * @return Filename formatted as `{sample_index:09d}.jpg`.
 */
std::string image_filename(std::uint64_t sample_index) {
    std::ostringstream output;
    output << std::setfill('0') << std::setw(9) << sample_index << ".jpg";
    return output.str();
}

/**
 * @brief Calculates one target timestamp without signed overflow.
 *
 * @param sample_index Zero-based sampling ordinal.
 * @param sample_period_ms Positive configured sampling period in milliseconds.
 * @return Exact non-negative target timestamp in milliseconds.
 * @throws std::overflow_error If the timestamp cannot fit in int64_t.
 */
std::int64_t target_timestamp_for(
    std::uint64_t sample_index,
    std::int64_t sample_period_ms
) {
    const std::uint64_t period = static_cast<std::uint64_t>(sample_period_ms);
    const std::uint64_t maximum = static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max()
    );
    if (sample_index > maximum / period) {
        throw std::overflow_error("sample target timestamp exceeds int64");
    }
    return static_cast<std::int64_t>(sample_index * period);
}

/**
 * @brief Computes the number of 1-FPS targets strictly inside a known duration.
 *
 * @param duration_ms Measured non-negative duration, or -1 when unavailable.
 * @param sample_period_ms Positive configured target period.
 * @param emitted_frame_count Count emitted from actual decoded timestamps.
 * @return ceil(duration_ms / sample_period_ms), or emitted_frame_count when
 *         FFmpeg did not provide a stream duration.
 */
std::uint64_t expected_target_count(
    std::int64_t duration_ms,
    std::int64_t sample_period_ms,
    std::uint64_t emitted_frame_count
) {
    if (duration_ms < 0) {
        return emitted_frame_count;
    }

    const std::uint64_t duration = static_cast<std::uint64_t>(duration_ms);
    const std::uint64_t period = static_cast<std::uint64_t>(sample_period_ms);
    return (duration / period) + (duration % period == 0 ? 0U : 1U);
}

/**
 * @brief Tests whether a state already owns a retained native extraction result.
 *
 * These states must not be retried by Task 6 because downstream enrichment or
 * publication may already rely on the exact staging/source artifact identity.
 *
 * @param status Current durable state status.
 * @return True when native extraction must skip rather than delete/recreate data.
 */
bool retains_native_bundle(VideoStatus status) {
    return status == VideoStatus::EnrichmentPending ||
           status == VideoStatus::Enriched ||
           status == VideoStatus::Published ||
           status == VideoStatus::Cleaned;
}

/**
 * @brief Verifies that a state belongs to the current immutable run identity.
 *
 * @param state Parsed state to inspect.
 * @param identity Expected run, video, extractor-version, and config identity.
 * @return None; returns only when every immutable provenance value matches.
 * @throws std::invalid_argument If state provenance does not match identity.
 */
void validate_matching_identity(
    const VideoState& state,
    const StateIdentity& identity
) {
    if (state.run_id != identity.run_id ||
        state.video_id != identity.video_id ||
        state.extractor_version != identity.extractor_version ||
        state.config_hash != identity.config_hash) {
        throw std::invalid_argument("state provenance does not match extraction request");
    }
}

/**
 * @brief Reads an existing state or publishes the first pending checkpoint.
 *
 * @param state_path Final per-video state JSON path.
 * @param identity Immutable expected provenance for the state.
 * @param input Source manifest row whose URL is retained on first creation.
 * @return Existing matching state or a newly persisted pending state.
 * @throws std::runtime_error If the state cannot be read or written.
 * @throws std::invalid_argument If existing provenance does not match.
 */
VideoState load_or_create_state(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    const VideoInput& input
) {
    if (!path_exists(state_path)) {
        VideoState state = make_pending_state(identity, input.watch_url);
        save_state_atomic(state_path, state);
        return state;
    }

    VideoState state = read_state(state_path);
    validate_matching_identity(state, identity);
    return state;
}

/**
 * @brief Removes only retryable stale artifacts owned by one validated video ID.
 *
 * @param run_root Native run root containing source and staging directories.
 * @param video_id Validated canonical video identifier.
 * @return None; stale staging, final source, copy temp, and yt-dlp outputs are absent.
 * @throws std::system_error If an exact stale artifact cannot be inspected or removed.
 * @throws std::runtime_error If a matching downloader artifact is not a regular file.
 */
void clear_stale_video_artifacts(
    const std::filesystem::path& run_root,
    const std::string& video_id
) {
    const std::filesystem::path staging_directory =
        run_root / "staging" / video_id;
    const std::filesystem::path source_directory = run_root / "source";
    const std::filesystem::path source_path =
        source_directory / (video_id + ".part");
    std::filesystem::path source_temporary_path = source_path;
    source_temporary_path += ".tmp";

    remove_staging_directory(staging_directory);
    remove_exact_path(source_path);
    remove_exact_path(source_temporary_path);

    if (!path_exists(source_directory)) {
        return;
    }

    const std::string download_prefix = video_id + ".download.";
    std::error_code iteration_error;
    std::filesystem::directory_iterator iterator(source_directory, iteration_error);
    if (iteration_error) {
        throw std::system_error(
            iteration_error,
            "inspect source directory " + source_directory.string()
        );
    }

    const std::filesystem::directory_iterator end;
    while (iterator != end) {
        const std::filesystem::directory_entry entry = *iterator;
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(download_prefix, 0) == 0) {
            std::error_code type_error;
            if (!entry.is_regular_file(type_error)) {
                if (type_error) {
                    throw std::system_error(
                        type_error,
                        "inspect downloader output " + entry.path().string()
                    );
                }
                throw std::runtime_error(
                    "matching downloader output is not a regular file: " +
                    entry.path().string()
                );
            }
            remove_exact_path(entry.path());
        }

        iterator.increment(iteration_error);
        if (iteration_error) {
            throw std::system_error(
                iteration_error,
                "iterate source directory " + source_directory.string()
            );
        }
    }
}

/**
 * @brief Copies an offline source fixture through a same-directory temporary file.
 *
 * @param source_root Directory containing `{video_id}.mp4` test fixtures.
 * @param video_id Validated canonical video identifier.
 * @param destination Final `source/{video_id}.part` destination path.
 * @return None; destination is a complete local source video after success.
 * @throws std::runtime_error If the fixture is absent or not a regular file.
 * @throws std::system_error If copying or final rename fails.
 */
void copy_local_source(
    const std::filesystem::path& source_root,
    const std::string& video_id,
    const std::filesystem::path& destination
) {
    const std::filesystem::path fixture = source_root / (video_id + ".mp4");
    std::error_code error;
    if (!std::filesystem::is_regular_file(fixture, error)) {
        if (error) {
            throw std::system_error(error, "inspect local source " + fixture.string());
        }
        throw std::runtime_error("local source fixture is missing: " + fixture.string());
    }

    ensure_directory(destination.parent_path());
    std::filesystem::path temporary_path = destination;
    temporary_path += ".tmp";

    std::filesystem::copy_file(
        fixture,
        temporary_path,
        std::filesystem::copy_options::overwrite_existing,
        error
    );
    if (error) {
        throw std::system_error(
            error,
            "copy local source " + fixture.string() + " to " + temporary_path.string()
        );
    }

    publish_by_rename(temporary_path, destination);
}

/**
 * @brief Formats a bounded child-process diagnostic for durable failed state.
 *
 * @param result Completed downloader process result.
 * @return Non-blank diagnostic containing exit/signal status and child output.
 */
std::string process_failure_message(const ProcessResult& result) {
    std::ostringstream output;
    output << "yt-dlp failed";
    if (result.signal_number != 0) {
        output << " with signal " << result.signal_number;
    } else {
        output << " with exit code " << result.exit_code;
    }

    if (!result.stderr_text.empty()) {
        output << ": " << result.stderr_text;
    } else if (!result.stdout_text.empty()) {
        output << ": " << result.stdout_text;
    }
    return output.str();
}

/**
 * @brief Finds the single completed yt-dlp output prepared for one video.
 *
 * @param source_directory Directory that received the deterministic output template.
 * @param video_id Validated canonical video identifier.
 * @return The sole regular file whose name begins `{video_id}.download.`.
 * @throws std::runtime_error If zero or multiple candidate files exist.
 * @throws std::system_error If the source directory cannot be inspected.
 */
std::filesystem::path find_downloaded_source(
    const std::filesystem::path& source_directory,
    const std::string& video_id
) {
    const std::string download_prefix = video_id + ".download.";
    std::vector<std::filesystem::path> candidates;
    std::error_code iteration_error;
    std::filesystem::directory_iterator iterator(source_directory, iteration_error);
    if (iteration_error) {
        throw std::system_error(
            iteration_error,
            "inspect downloader output directory " + source_directory.string()
        );
    }

    const std::filesystem::directory_iterator end;
    while (iterator != end) {
        const std::filesystem::directory_entry entry = *iterator;
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(download_prefix, 0) == 0) {
            std::error_code type_error;
            if (!entry.is_regular_file(type_error)) {
                if (type_error) {
                    throw std::system_error(
                        type_error,
                        "inspect downloader output " + entry.path().string()
                    );
                }
                throw std::runtime_error(
                    "downloader output is not a regular file: " +
                    entry.path().string()
                );
            }
            candidates.push_back(entry.path());
        }

        iterator.increment(iteration_error);
        if (iteration_error) {
            throw std::system_error(
                iteration_error,
                "iterate downloader output directory " + source_directory.string()
            );
        }
    }

    if (candidates.size() != 1) {
        throw std::runtime_error(
            "expected exactly one yt-dlp output for " + video_id +
            ", found " + std::to_string(candidates.size())
        );
    }
    return candidates.front();
}

/**
 * @brief Downloads one network source with explicit shell-free yt-dlp arguments.
 *
 * @param input Manifest row containing the literal watch URL.
 * @param config Active extractor configuration including yt-dlp executable.
 * @param destination Final `source/{video_id}.part` destination path.
 * @return None; destination is a complete downloaded video after success.
 * @throws std::runtime_error If yt-dlp fails or does not emit exactly one source.
 * @throws std::system_error If output directories or final rename cannot be prepared.
 */
void download_source(
    const VideoInput& input,
    const ExtractionConfig& config,
    const std::filesystem::path& destination
) {
    const std::filesystem::path source_directory = destination.parent_path();
    ensure_directory(source_directory);
    const std::filesystem::path output_template =
        source_directory / (input.video_id + ".download.%(ext)s");

    std::vector<std::string> arguments{
        config.yt_dlp_binary,
    };
    if (config.yt_dlp_cookies_path.has_value()) {
        arguments.push_back("--cookies");
        arguments.push_back(config.yt_dlp_cookies_path.value());
    }
    if (config.yt_dlp_js_runtime.has_value()) {
        arguments.push_back("--js-runtimes");
        arguments.push_back(config.yt_dlp_js_runtime.value());
    }
    const std::vector<std::string> download_arguments{
        "--no-playlist",
        "--no-progress",
        "--no-part",
        "--format",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "--output",
        output_template.string(),
        input.watch_url,
    };
    arguments.insert(
        arguments.end(),
        download_arguments.begin(),
        download_arguments.end()
    );
    const ProcessResult result = run_process(arguments);
    if (result.exit_code != 0 || result.signal_number != 0) {
        throw std::runtime_error(process_failure_message(result));
    }

    const std::filesystem::path downloaded = find_downloaded_source(
        source_directory,
        input.video_id
    );
    publish_by_rename(downloaded, destination);
}

/**
 * @brief Prepares one local source video through test copy or yt-dlp download.
 *
 * @param input Manifest row whose video ID and watch URL define the source.
 * @param config Active extractor configuration.
 * @param run_root Native run root that owns the source directory.
 * @param source_root Optional offline fixture directory used only for local runs.
 * @return Complete `source/{video_id}.part` path ready for FFmpeg decoding.
 * @throws std::runtime_error If the requested source cannot be prepared.
 * @throws std::system_error If required source filesystem operations fail.
 */
std::filesystem::path prepare_source(
    const VideoInput& input,
    const ExtractionConfig& config,
    const std::filesystem::path& run_root,
    const std::optional<std::filesystem::path>& source_root
) {
    const std::filesystem::path destination =
        run_root / "source" / (input.video_id + ".part");
    if (source_root.has_value()) {
        copy_local_source(source_root.value(), input.video_id, destination);
    } else {
        download_source(input, config, destination);
    }
    return destination;
}

/**
 * @brief Selects the decoded image identified by one sampler selection.
 *
 * TimestampSampler can only select the current or immediately previous frame.
 * Keeping this check explicit prevents a future sampler change from silently
 * retaining more decoded frames than the bounded-memory extraction contract.
 *
 * @param selection Sampler result identifying the chosen decoded ordinal.
 * @param previous Optional immediately preceding decoded frame.
 * @param current Current decoded frame supplied to TimestampSampler.
 * @return Reference to the selected decoded frame image and timing metadata.
 * @throws std::logic_error If selection refers to neither previous nor current.
 */
const DecodedFrame& selected_decoded_frame(
    const SelectedTarget& selection,
    const std::optional<DecodedFrame>& previous,
    const DecodedFrame& current
) {
    if (selection.selected_ordinal == current.ordinal) {
        return current;
    }
    if (previous.has_value() &&
        selection.selected_ordinal == previous->ordinal) {
        return previous.value();
    }
    throw std::logic_error(
        "timestamp sampler selected a frame outside previous/current retention"
    );
}

/**
 * @brief Writes one selected frame's durable and optional enrichment images.
 *
 * @param input Manifest row that owns the emitted frame identity.
 * @param config Active JPEG and sampling configuration.
 * @param video_info FFmpeg stream metadata retained in the native row.
 * @param bundle_directory Per-video staging bundle root.
 * @param selection Target timestamp and selected decoded ordinal.
 * @param frame Decoded frame selected by TimestampSampler.
 * @param output Open frames JSONL temporary stream.
 * @return Fully populated emitted NativeFrameRow for validation accounting.
 * @throws std::runtime_error If image encoding or JSONL writing fails.
 * @throws std::invalid_argument If decoded/frame metadata violates contracts.
 */
NativeFrameRow emit_selected_frame(
    const VideoInput& input,
    const ExtractionConfig& config,
    const VideoInfo& video_info,
    const std::filesystem::path& bundle_directory,
    const SelectedTarget& selection,
    const DecodedFrame& frame,
    std::ostream& output
) {
    if (!frame.image) {
        throw std::runtime_error("selected decoded frame has no image data");
    }

    const std::string filename = image_filename(selection.sample_index);
    const std::filesystem::path durable_path =
        bundle_directory / "images" / filename;
    const EncodedImage durable_image = encode_jpeg(
        *frame.image,
        durable_path,
        ImageVariant{
            config.durable_long_edge,
            config.durable_jpeg_quality,
        }
    );

    std::string enrichment_image_path;
    std::uint64_t enrichment_image_size_bytes = 0;
    if (config.write_enrichment_images) {
        const std::filesystem::path enrichment_path =
            bundle_directory / "enrichment_images" / filename;
        const EncodedImage enrichment_image = encode_jpeg(
            *frame.image,
            enrichment_path,
            ImageVariant{
                0,
                config.enrichment_jpeg_quality,
            }
        );
        enrichment_image_path = (
            std::filesystem::path("enrichment_images") / filename
        ).generic_string();
        enrichment_image_size_bytes = enrichment_image.bytes;
    }

    NativeFrameRow row{
        make_frame_id(input.video_id, selection.sample_index),
        input.video_id,
        selection.sample_index,
        selection.target_timestamp_ms,
        frame.timestamp_ms,
        submission_frame_idx(video_info.avg_fps, frame.timestamp_ms),
        video_info.avg_fps,
        video_info.avg_fps_rational,
        frame.pts,
        video_info.time_base,
        durable_image.width,
        durable_image.height,
        (std::filesystem::path("images") / filename).generic_string(),
        std::move(enrichment_image_path),
        durable_image.bytes,
        enrichment_image_size_bytes,
    };
    write_frame_row(output, row);
    return row;
}

/**
 * @brief Returns one JPEG file's byte count as a checked uint64 value.
 *
 * @param path Existing regular file whose size is required for bundle validation.
 * @return Exact non-negative byte size of path.
 * @throws std::runtime_error If path is missing or not a regular file.
 * @throws std::system_error If file metadata cannot be read.
 * @throws std::overflow_error If size does not fit in uint64_t.
 */
std::uint64_t checked_file_size(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        if (error) {
            throw std::system_error(error, "inspect frame image " + path.string());
        }
        throw std::runtime_error("frame image is missing: " + path.string());
    }

    const std::uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error) {
        throw std::system_error(error, "read frame image size " + path.string());
    }
    if (bytes > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("frame image size exceeds uint64: " + path.string());
    }
    return static_cast<std::uint64_t>(bytes);
}

/**
 * @brief Validates emitted JSONL rows and their referenced per-video images.
 *
 * @param input Manifest row expected to own every native frame row.
 * @param config Active extraction settings that control optional enrichment paths.
 * @param video_info FFmpeg stream metadata used to recompute frame_idx.
 * @param bundle_directory Per-video staging root containing JSONL and images.
 * @param frames_path Temporary or final JSONL path to validate.
 * @param expected_count Exact expected number of frame rows.
 * @return None; returns only when identity, order, paths, sizes, and frame_idx match.
 * @throws std::invalid_argument If parsed rows violate native artifact contracts.
 * @throws std::runtime_error If rows/images do not match the emitted bundle.
 */
void validate_emitted_bundle(
    const VideoInput& input,
    const ExtractionConfig& config,
    const VideoInfo& video_info,
    const std::filesystem::path& bundle_directory,
    const std::filesystem::path& frames_path,
    std::uint64_t expected_count
) {
    const std::vector<NativeFrameRow> rows = read_frame_jsonl(frames_path);
    if (rows.size() != expected_count) {
        throw std::runtime_error(
            "native frame JSONL count mismatch for " + input.video_id +
            ": expected " + std::to_string(expected_count) + ", found " +
            std::to_string(rows.size())
        );
    }

    for (std::size_t index = 0; index < rows.size(); ++index) {
        const NativeFrameRow& row = rows[index];
        if (row.video_id != input.video_id) {
            throw std::runtime_error("native frame row video_id mismatch");
        }
        if (row.sample_index != static_cast<std::uint64_t>(index)) {
            throw std::runtime_error("native frame row sample order mismatch");
        }
        if (row.frame_id != make_frame_id(input.video_id, row.sample_index)) {
            throw std::runtime_error("native frame row frame_id mismatch");
        }
        if (row.target_timestamp_ms !=
            target_timestamp_for(row.sample_index, config.sample_period_ms)) {
            throw std::runtime_error("native frame row target timestamp mismatch");
        }
        if (row.frame_idx != submission_frame_idx(video_info.avg_fps, row.timestamp_ms)) {
            throw std::runtime_error("native frame row frame_idx mismatch");
        }
        if (row.avg_fps_rational.numerator != video_info.avg_fps_rational.numerator ||
            row.avg_fps_rational.denominator != video_info.avg_fps_rational.denominator ||
            row.time_base.numerator != video_info.time_base.numerator ||
            row.time_base.denominator != video_info.time_base.denominator) {
            throw std::runtime_error("native frame row stream metadata mismatch");
        }

        const std::string filename = image_filename(row.sample_index);
        if (row.image_path !=
            (std::filesystem::path("images") / filename).generic_string()) {
            throw std::runtime_error("native frame row durable path mismatch");
        }
        if (checked_file_size(bundle_directory / row.image_path) !=
            row.image_size_bytes) {
            throw std::runtime_error("native frame row durable byte-size mismatch");
        }

        const std::string expected_enrichment_path = (
            std::filesystem::path("enrichment_images") / filename
        ).generic_string();
        if (config.write_enrichment_images) {
            if (row.enrichment_image_path != expected_enrichment_path) {
                throw std::runtime_error("native frame row enrichment path mismatch");
            }
            if (checked_file_size(bundle_directory / row.enrichment_image_path) !=
                row.enrichment_image_size_bytes) {
                throw std::runtime_error(
                    "native frame row enrichment byte-size mismatch"
                );
            }
        } else if (!row.enrichment_image_path.empty() ||
                   row.enrichment_image_size_bytes != 0) {
            throw std::runtime_error(
                "native frame row unexpectedly contains enrichment image metadata"
            );
        }
    }
}

/**
 * @brief Serializes a validated native per-video manifest through json-c.
 *
 * @param manifest Native metadata produced after frame JSONL validation.
 * @return Compact JSON document with escaped paths and exact numeric fields.
 * @throws std::invalid_argument If manifest fields violate extraction invariants.
 * @throws std::runtime_error If json-c cannot allocate a required object.
 */
std::string serialize_native_manifest(const NativeVideoManifest& manifest) {
    if (manifest.video_id.empty() || manifest.status.empty() ||
        manifest.extractor_version.empty() || manifest.config_hash.empty() ||
        manifest.frames_jsonl.empty() || manifest.duration_ms < -1 ||
        manifest.avg_fps_rational.denominator == 0 ||
        !std::isfinite(manifest.avg_fps) || manifest.avg_fps <= 0.0) {
        throw std::invalid_argument("native per-video manifest is invalid");
    }

    JsonObject object(json_object_new_object());
    if (object.get() == nullptr) {
        throw std::runtime_error("unable to allocate native per-video manifest");
    }

    add_json_string(object.get(), "video_id", manifest.video_id);
    add_json_string(object.get(), "status", manifest.status);
    add_json_integer(object.get(), "duration_ms", manifest.duration_ms);
    add_json_unsigned(
        object.get(),
        "expected_frame_count",
        manifest.expected_frame_count
    );
    add_json_unsigned(
        object.get(),
        "emitted_frame_count",
        manifest.emitted_frame_count
    );
    add_json_double(object.get(), "avg_fps", manifest.avg_fps);
    add_json_integer(
        object.get(),
        "avg_fps_num",
        manifest.avg_fps_rational.numerator
    );
    add_json_integer(
        object.get(),
        "avg_fps_den",
        manifest.avg_fps_rational.denominator
    );
    add_json_string(
        object.get(),
        "extractor_version",
        manifest.extractor_version
    );
    add_json_string(object.get(), "config_hash", manifest.config_hash);
    add_json_string(object.get(), "frames_jsonl", manifest.frames_jsonl);

    return json_object_to_json_string_ext(
        object.get(),
        JSON_C_TO_STRING_PLAIN
    );
}

/**
 * @brief Atomically writes one native manifest beside its staging frame JSONL.
 *
 * @param path Final `manifest.json` path in the already-created bundle directory.
 * @param manifest Valid per-video native manifest to serialize and publish.
 * @return None; readers see an old complete manifest or the new complete manifest.
 * @throws std::runtime_error If the temporary manifest cannot be written.
 * @throws std::system_error If the parent directory or final rename fails.
 */
void write_native_manifest_atomic(
    const std::filesystem::path& path,
    const NativeVideoManifest& manifest
) {
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument("native manifest path must name a file");
    }

    ensure_directory(path.parent_path());
    const std::string serialized = serialize_native_manifest(manifest);
    std::filesystem::path temporary_path = path;
    temporary_path += ".tmp";

    try {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "unable to open temporary native manifest: " +
                temporary_path.string()
            );
        }
        output << serialized;
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "unable to flush temporary native manifest: " +
                temporary_path.string()
            );
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "unable to close temporary native manifest: " +
                temporary_path.string()
            );
        }

        publish_by_rename(temporary_path, path);
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        throw;
    }
}

/**
 * @brief Decodes a prepared source and writes one fully validated staging bundle.
 *
 * @param input Manifest row that owns the bundle identity.
 * @param config Active sampling and JPEG configuration.
 * @param source_path Complete local source file prepared for FFmpeg.
 * @param bundle_directory Empty per-video staging directory for this attempt.
 * @return Native manifest describing the atomically published frame JSONL.
 * @throws std::runtime_error If decoding, image encoding, or validation fails.
 * @throws std::system_error If staging artifact filesystem operations fail.
 */
NativeVideoManifest extract_native_bundle(
    const VideoInput& input,
    const ExtractionConfig& config,
    const std::filesystem::path& source_path,
    const std::filesystem::path& bundle_directory
) {
    ensure_directory(bundle_directory / "images");
    if (config.write_enrichment_images) {
        ensure_directory(bundle_directory / "enrichment_images");
    }

    const std::filesystem::path temporary_frames_path =
        bundle_directory / "frames.jsonl.tmp";
    const std::filesystem::path frames_path = bundle_directory / "frames.jsonl";
    VideoDecoder decoder(source_path);
    const VideoInfo video_info = decoder.info();
    TimestampSampler sampler(config.sample_period_ms);
    std::optional<DecodedFrame> previous;
    std::uint64_t emitted_frame_count = 0;
    const DiskBudgetGuard disk_guard(
        static_cast<std::uint64_t>(config.disk_reserve_bytes)
    );
    const std::uint64_t estimated_frame_bytes = estimate_frame_write_bytes(
        config.durable_long_edge,
        config.write_enrichment_images
    );

    {
        std::ofstream output(
            temporary_frames_path,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "unable to open temporary native frames JSONL: " +
                temporary_frames_path.string()
            );
        }

        while (true) {
            std::optional<DecodedFrame> current = decoder.next();
            if (!current.has_value()) {
                break;
            }

            const std::vector<SelectedTarget> selections = sampler.push(
                TimedFrame{current->ordinal, current->timestamp_ms}
            );
            for (const SelectedTarget& selection : selections) {
                // Targets equal to stream duration are outside the half-open
                // [0, duration) timeline and must not become extra one-FPS rows.
                if (video_info.duration_ms >= 0 &&
                    selection.target_timestamp_ms >= video_info.duration_ms) {
                    continue;
                }

                const DecodedFrame& selected = selected_decoded_frame(
                    selection,
                    previous,
                    current.value()
                );
                // Refuse to start writing a frame that would breach the local
                // disk reserve; a partially written frame is never valid.
                disk_guard.require_capacity(bundle_directory, estimated_frame_bytes);
                static_cast<void>(emit_selected_frame(
                    input,
                    config,
                    video_info,
                    bundle_directory,
                    selection,
                    selected,
                    output
                ));
                if (emitted_frame_count ==
                    std::numeric_limits<std::uint64_t>::max()) {
                    throw std::overflow_error("native emitted frame count exceeds uint64");
                }
                ++emitted_frame_count;
            }

            previous = std::move(current);
        }

        output.flush();
        if (!output) {
            throw std::runtime_error(
                "unable to flush temporary native frames JSONL: " +
                temporary_frames_path.string()
            );
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "unable to close temporary native frames JSONL: " +
                temporary_frames_path.string()
            );
        }
    }

    const std::uint64_t expected_frame_count = expected_target_count(
        video_info.duration_ms,
        config.sample_period_ms,
        emitted_frame_count
    );
    if (expected_frame_count != emitted_frame_count) {
        throw std::runtime_error(
            "decoded sample count mismatch for " + input.video_id +
            ": expected " + std::to_string(expected_frame_count) +
            ", emitted " + std::to_string(emitted_frame_count)
        );
    }
    validate_emitted_bundle(
        input,
        config,
        video_info,
        bundle_directory,
        temporary_frames_path,
        emitted_frame_count
    );
    publish_by_rename(temporary_frames_path, frames_path);

    NativeVideoManifest manifest{
        input.video_id,
        "enrichment_pending",
        video_info.duration_ms,
        expected_frame_count,
        emitted_frame_count,
        video_info.avg_fps,
        video_info.avg_fps_rational,
        config.extractor_version,
        config.config_hash,
        "frames.jsonl",
    };
    write_native_manifest_atomic(bundle_directory / "manifest.json", manifest);
    return manifest;
}

/**
 * @brief Moves an interrupted active state into failed before a clean retry.
 *
 * @param state_path Final per-video state JSON path.
 * @param identity Immutable expected state provenance.
 * @param state Current matching state read from state_path.
 * @return Failed state for interrupted attempts, otherwise the original state.
 * @throws std::invalid_argument If state cannot make a valid retry transition.
 * @throws std::runtime_error If state persistence fails.
 */
VideoState fail_interrupted_attempt(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    const VideoState& state
) {
    if (state.status == VideoStatus::Pending || state.status == VideoStatus::Failed) {
        return state;
    }
    return transition_state(
        state_path,
        identity,
        state.status,
        VideoStatus::Failed,
        "restart interrupted native extraction from clean per-video staging"
    );
}

/**
 * @brief Starts a clean source acquisition attempt from pending or failed state.
 *
 * @param state_path Final per-video state JSON path.
 * @param identity Immutable expected state provenance.
 * @param state Current matching state, possibly interrupted before this call.
 * @param run_root Native run root whose exact per-video retry artifacts are cleared.
 * @return Persisted downloading state with stale mutable extraction fields cleared.
 * @throws std::runtime_error If stale artifact cleanup or state persistence fails.
 */
VideoState start_clean_attempt(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    VideoState state,
    const std::filesystem::path& run_root
) {
    state = fail_interrupted_attempt(state_path, identity, state);
    clear_stale_video_artifacts(run_root, identity.video_id);
    state = transition_state(
        state_path,
        identity,
        state.status,
        VideoStatus::Downloading
    );

    state.source_path.clear();
    state.last_completed_sample_index.reset();
    state.emitted_frame_count = 0;
    state.native_manifest_path.clear();
    state.enrichment_manifest_path.clear();
    save_state_atomic(state_path, state);
    return state;
}

/**
 * @brief Extracts one manifest video or skips its retained native bundle.
 *
 * @param input Selected manifest row.
 * @param config Active extraction configuration.
 * @param run_id Stable run ID derived from run_root.
 * @param run_root Native storage root for state, source, and staging outputs.
 * @param source_root Optional offline fixture directory.
 * @return Emitted frame count after success, or std::nullopt for a retained skip.
 * @throws std::runtime_error If source preparation, decode, or validation fails.
 * @throws std::invalid_argument If state provenance/lifecycle is incompatible.
 */
std::optional<std::uint64_t> extract_one_video(
    const VideoInput& input,
    const ExtractionConfig& config,
    const std::string& run_id,
    const std::filesystem::path& run_root,
    const std::optional<std::filesystem::path>& source_root
) {
    const StateIdentity identity{
        run_id,
        input.video_id,
        config.extractor_version,
        config.config_hash,
    };
    const std::filesystem::path state_path =
        run_root / "state" / (input.video_id + ".json");
    VideoState state = load_or_create_state(state_path, identity, input);
    if (retains_native_bundle(state.status)) {
        return std::nullopt;
    }

    state = start_clean_attempt(state_path, identity, std::move(state), run_root);
    const std::filesystem::path source_path = prepare_source(
        input,
        config,
        run_root,
        source_root
    );
    state.source_path = source_relative_path(input.video_id);
    save_state_atomic(state_path, state);
    state = transition_state(
        state_path,
        identity,
        VideoStatus::Downloading,
        VideoStatus::Extracting
    );

    const NativeVideoManifest manifest = extract_native_bundle(
        input,
        config,
        source_path,
        run_root / "staging" / input.video_id
    );
    state.last_completed_sample_index = manifest.emitted_frame_count == 0
        ? std::nullopt
        : std::optional<std::uint64_t>(manifest.emitted_frame_count - 1);
    state.emitted_frame_count = manifest.emitted_frame_count;
    state.native_manifest_path = native_manifest_relative_path(input.video_id);
    save_state_atomic(state_path, state);
    state = transition_state(
        state_path,
        identity,
        VideoStatus::Extracting,
        VideoStatus::Extracted
    );
    static_cast<void>(transition_state(
        state_path,
        identity,
        VideoStatus::Extracted,
        VideoStatus::EnrichmentPending
    ));
    return manifest.emitted_frame_count;
}

/**
 * @brief Converts an exception into a non-blank durable extraction diagnostic.
 *
 * @param error Exception raised by a source, decoder, image, or validation step.
 * @return Non-blank diagnostic suitable for VideoStatus::Failed state storage.
 */
std::string failure_message(const std::exception& error) {
    const std::string message = error.what();
    return message.empty() ? "native extraction failed" : message;
}

/**
 * @brief Best-effort persists a failure transition without masking the original error.
 *
 * @param state_path Per-video state JSON path that may already have been created.
 * @param identity Immutable expected state provenance.
 * @param diagnostic Non-blank source/decode/validation failure detail.
 * @return None; failures to record a failure state are intentionally suppressed.
 */
void record_failure_if_possible(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    const std::string& diagnostic
) noexcept {
    try {
        if (!path_exists(state_path)) {
            return;
        }

        const VideoState state = read_state(state_path);
        validate_matching_identity(state, identity);
        if (state.status == VideoStatus::Failed || retains_native_bundle(state.status)) {
            return;
        }
        static_cast<void>(transition_state(
            state_path,
            identity,
            state.status,
            VideoStatus::Failed,
            diagnostic.empty() ? "native extraction failed" : diagnostic
        ));
    } catch (...) {
        // The original failure remains the most actionable result when state
        // storage itself is unavailable or provenance has been rejected.
    }
}

/**
 * @brief Loads manifest rows and applies an optional one-video selection filter.
 *
 * @param manifest_path Input JSONL manifest path.
 * @param selected_video_id Optional canonical video ID requested by the CLI.
 * @return Ordered, non-empty selected manifest rows with safe video IDs.
 * @throws std::invalid_argument If the filter is blank/unknown or an ID is unsafe.
 * @throws std::runtime_error If the manifest cannot be opened.
 */
std::vector<VideoInput> select_videos(
    const std::filesystem::path& manifest_path,
    const std::optional<std::string>& selected_video_id
) {
    const std::vector<VideoInput> manifest = read_video_manifest(manifest_path);
    if (!selected_video_id.has_value()) {
        for (const VideoInput& input : manifest) {
            validate_video_id(input.video_id);
        }
        return manifest;
    }

    if (selected_video_id->empty()) {
        throw std::invalid_argument("video_id filter must not be blank");
    }
    validate_video_id(selected_video_id.value());
    for (const VideoInput& input : manifest) {
        if (input.video_id == selected_video_id.value()) {
            validate_video_id(input.video_id);
            return {input};
        }
    }
    throw std::invalid_argument(
        "requested video_id is absent from manifest: " + selected_video_id.value()
    );
}

/**
 * @brief Validates top-level extraction request paths before state mutation.
 *
 * @param request Caller-supplied extraction invocation settings.
 * @return None; returns only when all required paths are non-blank and safe.
 * @throws std::invalid_argument If a required path/filter is missing or invalid.
 */
void validate_request(const ExtractionRequest& request) {
    if (request.manifest_path.empty() || request.config_path.empty()) {
        throw std::invalid_argument("manifest and config paths must not be blank");
    }
    static_cast<void>(derive_run_id(request.run_root));
    if (request.video_id.has_value() && request.video_id->empty()) {
        throw std::invalid_argument("video_id filter must not be blank");
    }
    if (request.source_root.has_value() && request.source_root->empty()) {
        throw std::invalid_argument("source_root must not be blank when provided");
    }
}

}  // namespace

/**
 * @brief Executes bounded native extraction for all requested manifest rows.
 *
 * @param request Paths, optional video filter, source override, and failure policy.
 * @return Complete accounting for selected video outcomes and emitted frame rows.
 * @throws std::invalid_argument If top-level request/config/manifest input is invalid.
 * @throws std::runtime_error If top-level input or run-root setup cannot complete.
 */
ExtractionSummary extract_manifest(const ExtractionRequest& request) {
    validate_request(request);
    const ExtractionConfig config = read_extraction_config(request.config_path);
    const std::vector<VideoInput> selected = select_videos(
        request.manifest_path,
        request.video_id
    );
    ensure_directory(request.run_root);
    const std::string run_id = derive_run_id(request.run_root);

    ExtractionSummary summary;
    summary.pending = static_cast<std::uint64_t>(selected.size());
    for (const VideoInput& input : selected) {
        const StateIdentity identity{
            run_id,
            input.video_id,
            config.extractor_version,
            config.config_hash,
        };
        const std::filesystem::path state_path =
            request.run_root / "state" / (input.video_id + ".json");

        try {
            const std::optional<std::uint64_t> emitted = extract_one_video(
                input,
                config,
                run_id,
                request.run_root,
                request.source_root
            );
            if (emitted.has_value()) {
                ++summary.completed;
                summary.emitted_frame_count += emitted.value();
            } else {
                ++summary.skipped;
            }
            --summary.pending;
        } catch (const std::exception& error) {
            record_failure_if_possible(state_path, identity, failure_message(error));
            ++summary.failed;
            --summary.pending;
            if (request.fail_fast) {
                break;
            }
        } catch (...) {
            record_failure_if_possible(
                state_path,
                identity,
                "native extraction failed with a non-standard exception"
            );
            ++summary.failed;
            --summary.pending;
            if (request.fail_fast) {
                break;
            }
        }
    }
    return summary;
}

}  // namespace hcmai::keyframes_extraction
