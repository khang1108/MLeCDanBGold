/**
 * @file jsonl.hpp
 * @brief Declares strict JSONL readers and writers for native artifacts.
 *
 * This header owns transport between JSONL files and native contracts. It does
 * not decode video or construct FrameRecord/Parquet artifacts.
 */

#pragma once

#include "hcmai/keyframes_extraction/types.hpp"

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace hcmai::keyframes_extraction
{

    /**
     * @brief Reads the deterministic native video input manifest.
     *
     * @param path Path to the input JSONL file.
     * @return VideoInput rows in their original manifest order.
     * @throws std::runtime_error If path cannot be opened.
     * @throws std::invalid_argument If a row is malformed, blank, or duplicate.
     */
    std::vector<VideoInput> read_video_manifest(
        const std::filesystem::path &path);

    /**
     * @brief Reads validated native frame metadata rows from JSONL.
     *
     * @param path Path to the native per-video frame JSONL file.
     * @return NativeFrameRow values in the file's row order.
     * @throws std::runtime_error If path cannot be opened.
     * @throws std::invalid_argument If frame metadata is malformed or duplicate.
     */
    std::vector<NativeFrameRow> read_frame_jsonl(
        const std::filesystem::path &path);

    /**
     * @brief Serializes one validated native frame row as a JSONL line.
     *
     * @param output Open output stream that receives the JSON object and newline.
     * @param row Native frame metadata to validate and serialize.
     * @return None; writes exactly one JSONL row or throws.
     * @throws std::invalid_argument If row violates native artifact invariants.
     * @throws std::runtime_error If output cannot be written.
     */
    void write_frame_row(
        std::ostream &output,
        const NativeFrameRow &row);

    /**
     * @brief Writes a collection of native frame rows as JSONL.
     *
     * @param path Destination path; parent directories are created when needed.
     * @param rows Native frame metadata rows to serialize in vector order.
     * @return None; writes the JSONL file or throws.
     * @throws std::runtime_error If the destination cannot be created or written.
     * @throws std::invalid_argument If any row violates native artifact invariants.
     */
    void write_frame_jsonl(
        const std::filesystem::path &path,
        const std::vector<NativeFrameRow> &rows);

} // namespace hcmai::keyframes_extraction
