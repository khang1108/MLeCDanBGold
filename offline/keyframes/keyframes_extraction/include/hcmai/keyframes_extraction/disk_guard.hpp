/**
 * @file disk_guard.hpp
 * @brief Declares a conservative local-disk free-space reserve guard.
 *
 * This contract enforces the same disk admission invariant used by the
 * Python local A6000 pipeline: a planned write may proceed only if real free
 * bytes remain above a configured reserve afterward. It measures real
 * filesystem free space and never estimates from retained artifact sizes.
 */

#pragma once

#include <cstdint>
#include <filesystem>

namespace hcmai::keyframes_extraction
{

    /**
     * @brief Refuses a native write that would drop free disk space below a
     * fixed reserve.
     *
     * A reserve of zero disables enforcement, which preserves existing native
     * extraction behavior when the local pipeline's disk budget is not wired
     * into the active configuration.
     */
    class DiskBudgetGuard
    {
    public:
        /**
         * @param reserve_bytes Minimum free bytes that must remain on the
         *                       target filesystem after the planned write.
         */
        explicit DiskBudgetGuard(std::uint64_t reserve_bytes);

        /**
         * @brief Confirms writing `estimated_bytes` near `path` keeps real
         * free space at or above the configured reserve.
         *
         * @param path Any path on the target filesystem; need not yet exist.
         * @param estimated_bytes Conservative upper-bound byte estimate for
         *                        the planned write.
         * @throws std::runtime_error If free space would fall below the
         *                            configured reserve.
         * @throws std::system_error If free space cannot be measured.
         */
        void require_capacity(
            const std::filesystem::path &path,
            std::uint64_t estimated_bytes) const;

        /** @brief Returns the configured minimum free-byte reserve. */
        std::uint64_t reserve_bytes() const noexcept { return reserve_bytes_; }

    private:
        std::uint64_t reserve_bytes_;
    };

    /**
     * @brief Returns a conservative worst-case byte estimate for one durable
     * plus optional enrichment JPEG pair.
     *
     * The estimate deliberately over-counts actual JPEG output (which is
     * compressed well below this bound) so the reserve check never rejects a
     * write that would truly have succeeded.
     *
     * @param durable_long_edge Configured durable JPEG long edge in pixels.
     * @param write_enrichment_images Whether a second enrichment image is
     *                                 also written for the same frame.
     */
    std::uint64_t estimate_frame_write_bytes(
        int durable_long_edge,
        bool write_enrichment_images);

} // namespace hcmai::keyframes_extraction
