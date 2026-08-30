/**
 * @file disk_guard.cpp
 * @brief Implements the conservative local-disk free-space reserve guard.
 */

#include "hcmai/keyframes_extraction/disk_guard.hpp"

#include <stdexcept>
#include <system_error>

namespace hcmai::keyframes_extraction
{

    namespace
    {

        /**
         * @brief Walks up to the nearest existing ancestor of `path`.
         *
         * `std::filesystem::space` requires an existing path; a not-yet-created
         * staging directory must still resolve to a measurable filesystem.
         *
         * @param path Candidate path that may not exist yet.
         * @return The nearest existing ancestor, or the current path if none
         *         of `path`'s ancestors exist.
         */
        std::filesystem::path nearest_existing_ancestor(
            const std::filesystem::path &path)
        {
            std::error_code error;
            std::filesystem::path probe = path;
            while (!probe.empty() && !std::filesystem::exists(probe, error))
            {
                const std::filesystem::path parent = probe.parent_path();
                if (parent == probe)
                {
                    break;
                }
                probe = parent;
            }
            if (probe.empty() || !std::filesystem::exists(probe, error))
            {
                return std::filesystem::current_path();
            }
            return probe;
        }

    } // namespace

    DiskBudgetGuard::DiskBudgetGuard(std::uint64_t reserve_bytes)
        : reserve_bytes_(reserve_bytes)
    {
    }

    void DiskBudgetGuard::require_capacity(
        const std::filesystem::path &path,
        std::uint64_t estimated_bytes) const
    {
        const std::filesystem::path probe = nearest_existing_ancestor(path);
        std::error_code error;
        const std::filesystem::space_info info =
            std::filesystem::space(probe, error);
        if (error)
        {
            throw std::system_error(
                error, "measure free disk space for " + probe.string());
        }

        const std::uint64_t available = static_cast<std::uint64_t>(info.available);
        const bool would_underflow = available < estimated_bytes;
        const bool would_breach_reserve =
            !would_underflow && (available - estimated_bytes) < reserve_bytes_;
        if (would_underflow || would_breach_reserve)
        {
            throw std::runtime_error(
                "disk reserve exhausted: available=" +
                std::to_string(available) +
                " estimated_write=" + std::to_string(estimated_bytes) +
                " required_reserve=" + std::to_string(reserve_bytes_) +
                " path=" + probe.string());
        }
    }

    std::uint64_t estimate_frame_write_bytes(
        int durable_long_edge,
        bool write_enrichment_images)
    {
        // Assume an uncompressed 3-bytes-per-pixel square upper bound; real
        // MJPEG output is a small fraction of this at any configured quality.
        const std::uint64_t edge = static_cast<std::uint64_t>(
            durable_long_edge > 0 ? durable_long_edge : 1024);
        const std::uint64_t per_image_bytes = edge * edge * 3;
        return write_enrichment_images ? per_image_bytes * 2 : per_image_bytes;
    }

} // namespace hcmai::keyframes_extraction
