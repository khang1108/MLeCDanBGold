/**
 * @file extractor.hpp
 * @brief Declares one-video native extraction orchestration.
 *
 * This contract coordinates manifest selection, source preparation, FFmpeg
 * decoding, image emission, and checkpoint progression. It does not perform
 * downstream enrichment, publication, or corpus-level Parquet materialization.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace hcmai::keyframes_extraction {

/**
 * @brief Specifies one bounded native extraction invocation.
 *
 * A run identifier is derived deterministically from run_root so a state file
 * cannot be accidentally reused across differently named run directories.
 */
struct ExtractionRequest {
    /** @brief Input JSONL manifest containing canonical VideoInput rows. */
    std::filesystem::path manifest_path;
    /** @brief Root directory that owns state, source, staging, and later output. */
    std::filesystem::path run_root;
    /** @brief Strict JSON configuration path for this extraction invocation. */
    std::filesystem::path config_path;
    /** @brief Optional single video ID filter; absent processes all manifest rows. */
    std::optional<std::string> video_id;
    /** @brief Optional local `{video_id}.mp4` directory used by offline tests. */
    std::optional<std::filesystem::path> source_root;
    /** @brief Whether processing stops immediately after the first video failure. */
    bool fail_fast = false;
};

/**
 * @brief Aggregates outcomes from one manifest extraction invocation.
 */
struct ExtractionSummary {
    /** @brief Number of selected videos that reached enrichment_pending. */
    std::uint64_t completed = 0;
    /** @brief Number of selected videos that failed native source/extraction work. */
    std::uint64_t failed = 0;
    /** @brief Number of selected videos already retained at a later lifecycle state. */
    std::uint64_t skipped = 0;
    /** @brief Number of selected videos not attempted because fail_fast stopped work. */
    std::uint64_t pending = 0;
    /** @brief Total durable frame rows emitted by videos completed in this call. */
    std::uint64_t emitted_frame_count = 0;
};

/**
 * @brief Extracts all selected manifest videos into validated native bundles.
 *
 * Each successful video retains its source and staging bundle for the later
 * enrichment command. Failed videos retain diagnostics and partial staging;
 * a future invocation retries at video granularity from clean per-video paths.
 *
 * @param request Paths, optional video filter, and failure policy for this run.
 * @return Counts of completed, failed, skipped, pending, and emitted frames.
 * @throws std::invalid_argument If request paths, filter, configuration, or
 *                               manifest values violate native contracts.
 * @throws std::runtime_error If top-level input cannot be read or a required
 *                            run directory cannot be prepared.
 */
ExtractionSummary extract_manifest(const ExtractionRequest& request);

}  // namespace hcmai::keyframes_extraction
