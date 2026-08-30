/**
 * @file state.hpp
 * @brief Declares atomic per-video checkpoint state and lifecycle guards.
 *
 * This contract persists native extraction progress as JSON, validates allowed
 * status transitions, and exposes guarded enrichment/publication cleanup
 * commands. It does not download media or run enrichment models.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace hcmai::keyframes_extraction {

/**
 * @brief Maximum number of diagnostic bytes persisted in a failed state.
 *
 * Longer downloader or decoder diagnostics are truncated before state JSON is
 * published, preventing a single failure from creating an unbounded checkpoint.
 */
inline constexpr std::size_t kMaxStoredStateErrorBytes = 64U * 1024U;

/**
 * @brief Enumerates the durable per-video native extraction lifecycle.
 */
enum class VideoStatus {
    /** @brief State exists but source acquisition has not begun. */
    Pending,
    /** @brief yt-dlp or an offline source copy is in progress. */
    Downloading,
    /** @brief FFmpeg sampling and image encoding are in progress. */
    Extracting,
    /** @brief Native frames and manifest have been fully written. */
    Extracted,
    /** @brief Native bundle awaits validated enrichment handoff. */
    EnrichmentPending,
    /** @brief Enrichment handoff has been accepted by the native runner. */
    Enriched,
    /** @brief Durable native bundle has been published for materialization. */
    Published,
    /** @brief Source and temporary enrichment images have been safely removed. */
    Cleaned,
    /** @brief A recoverable native lifecycle stage failed with a diagnostic. */
    Failed,
};

/**
 * @brief Identifies the immutable provenance required to mutate a video state.
 *
 * These values prevent an old state file from being reused under a different
 * run, source video, extractor version, or extraction configuration.
 */
struct StateIdentity {
    /** @brief Native run identifier that owns the state file. */
    std::string run_id;
    /** @brief Canonical source-video identifier represented by the state file. */
    std::string video_id;
    /** @brief Native extractor version expected to own the state file. */
    std::string extractor_version;
    /** @brief Deterministic hash of the expected extraction configuration. */
    std::string config_hash;
};

/**
 * @brief Stores one serializable, durable checkpoint for a source video.
 *
 * All fields map directly to top-level JSON fields so Python and later native
 * commands can inspect provenance without reconstructing nested contracts.
 */
struct VideoState {
    /** @brief Native run identifier retained across every lifecycle transition. */
    std::string run_id;
    /** @brief Canonical source-video identifier represented by this state. */
    std::string video_id;
    /** @brief Original literal watch URL used for source acquisition. */
    std::string watch_url;
    /** @brief Local source path, blank until a source video has been prepared. */
    std::string source_path;
    /** @brief Native extractor version that created this state. */
    std::string extractor_version;
    /** @brief Deterministic extraction configuration hash. */
    std::string config_hash;
    /** @brief Current durable lifecycle status. */
    VideoStatus status = VideoStatus::Pending;
    /** @brief UTC creation timestamp in ISO-8601 `YYYY-MM-DDTHH:MM:SSZ` form. */
    std::string started_at;
    /** @brief UTC timestamp of the most recent successful state transition. */
    std::string updated_at;
    /** @brief Latest completed sample index, or null before any sample completes. */
    std::optional<std::uint64_t> last_completed_sample_index;
    /** @brief Count of native frame rows durably emitted so far. */
    std::uint64_t emitted_frame_count = 0;
    /** @brief Native per-video manifest path, blank until extraction completes. */
    std::string native_manifest_path;
    /** @brief Validated enrichment handoff path, blank until enrichment completes. */
    std::string enrichment_manifest_path;
    /** @brief Bounded failure diagnostic; blank for every non-failed status. */
    std::string error;
};

/**
 * @brief Builds the initial pending state for one immutable video provenance.
 *
 * @param identity Run, video, extractor-version, and configuration provenance.
 * @param watch_url Literal source watch URL to retain in the state file.
 * @return A validated VideoState with pending status and matching timestamps.
 * @throws std::invalid_argument If provenance or watch_url is blank.
 */
VideoState make_pending_state(
    const StateIdentity& identity,
    std::string watch_url
);

/**
 * @brief Reads and validates a durable video checkpoint JSON document.
 *
 * @param state_path Path to the final per-video state JSON file.
 * @return A fully validated VideoState value reconstructed from state_path.
 * @throws std::runtime_error If state_path cannot be opened.
 * @throws std::invalid_argument If JSON or serialized state values are invalid.
 */
VideoState read_state(const std::filesystem::path& state_path);

/**
 * @brief Writes a validated state through a same-directory temporary JSON file.
 *
 * The temporary path is `<state_path>.tmp`; after flush and close it is renamed
 * onto state_path, so readers observe either the old complete JSON or the new
 * complete JSON and no successful write leaves the temporary file behind.
 *
 * @param state_path Final state JSON destination path.
 * @param state Fully populated state value to serialize.
 * @return None; publishes the state atomically or throws without partial JSON.
 * @throws std::invalid_argument If state_path or state violates invariants.
 * @throws std::runtime_error If directories or the temporary file cannot be written.
 * @throws std::system_error If the final filesystem rename fails.
 */
void save_state_atomic(
    const std::filesystem::path& state_path,
    const VideoState& state
);

/**
 * @brief Validates provenance and advances one durable state lifecycle edge.
 *
 * The persisted state must match identity exactly and be in expected_status.
 * Only the documented forward edges, an active-state failure edge, and a
 * failed-to-downloading retry are accepted. A rejected request leaves the
 * current final JSON unchanged.
 *
 * @param state_path Final state JSON path to read and atomically replace.
 * @param identity Expected immutable run, video, version, and config identity.
 * @param expected_status Required current status before the transition.
 * @param next_status Requested successor lifecycle status.
 * @param error Failure diagnostic required only when next_status is Failed.
 * @return The state after its successful persisted transition.
 * @throws std::invalid_argument If provenance, status, edge, or error is invalid.
 * @throws std::runtime_error If state_path cannot be read or persisted.
 * @throws std::system_error If atomic replacement fails.
 */
VideoState transition_state(
    const std::filesystem::path& state_path,
    const StateIdentity& identity,
    VideoStatus expected_status,
    VideoStatus next_status,
    std::string error = {}
);

/**
 * @brief Accepts one validated enrichment handoff for an extracted video.
 *
 * The handoff must describe the same native staging manifest, video,
 * configuration hash, and emitted frame count as the current
 * ``enrichment_pending`` state. Repeating the command with the same handoff
 * is a no-op; a different handoff never overwrites accepted provenance.
 *
 * @param run_root Root directory containing the native state and staging bundle.
 * @param video_id Canonical video identifier whose state is advanced.
 * @param handoff_path Compact JSON handoff produced after specialist enrichment.
 * @return None; persists the enriched state or throws without partial state.
 * @throws std::invalid_argument If paths, handoff contents, or predecessor state
 *         violate the native lifecycle contract.
 * @throws std::runtime_error If required state or manifest files cannot be read.
 * @throws std::system_error If the updated checkpoint cannot be published.
 */
void mark_video_enriched(
    const std::filesystem::path& run_root,
    const std::string& video_id,
    const std::filesystem::path& handoff_path
);

/**
 * @brief Publishes a fully enriched staging bundle through a final manifest marker.
 *
 * Durable images, frame JSONL, and accepted enrichment provenance move together
 * from ``staging/{video_id}`` to ``published/{video_id}``. The published native
 * manifest is written last, so readers never treat a partial directory as a
 * complete bundle. A pre-existing complete published bundle is restored if the
 * replacement path fails before the state transition succeeds.
 *
 * @param run_root Root directory containing native state, staging, and published data.
 * @param video_id Canonical video identifier whose bundle is published.
 * @param manifest_path Native staging manifest supplied as an explicit guard.
 * @return None; persists the published state after the final manifest is visible.
 * @throws std::invalid_argument If provenance, paths, manifests, or lifecycle
 *         predecessor requirements do not match.
 * @throws std::runtime_error If required bundle artifacts cannot be read or moved.
 * @throws std::system_error If filesystem publication or rollback operations fail.
 */
void mark_video_published(
    const std::filesystem::path& run_root,
    const std::string& video_id,
    const std::filesystem::path& manifest_path
);

/**
 * @brief Removes only temporary source and staging artifacts after publication.
 *
 * The command intentionally retains ``published/{video_id}``, including durable
 * JPEGs and its final native manifest. It is idempotent after a state reaches
 * ``cleaned`` and never traverses another video's paths.
 *
 * @param run_root Root directory containing the native lifecycle directories.
 * @param video_id Canonical video identifier whose temporary files are removed.
 * @return None; persists the cleaned state after scoped cleanup succeeds.
 * @throws std::invalid_argument If video identity or predecessor state is invalid.
 * @throws std::runtime_error If the published manifest no longer validates.
 * @throws std::system_error If an exact temporary artifact cannot be removed.
 */
void cleanup_video(
    const std::filesystem::path& run_root,
    const std::string& video_id
);

}  // namespace hcmai::keyframes_extraction
