/**
 * @file test_disk_guard.cpp
 * @brief Unit tests for the conservative local-disk free-space reserve guard.
 */

#include "hcmai/keyframes_extraction/disk_guard.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

    using hcmai::keyframes_extraction::DiskBudgetGuard;
    using hcmai::keyframes_extraction::estimate_frame_write_bytes;
    using namespace hcmai::keyframes_extraction::test_support;

    void test_zero_reserve_allows_tiny_write()
    {
        const DiskBudgetGuard guard(0);
        const std::filesystem::path here = std::filesystem::current_path();
        guard.require_capacity(here, 1);
    }

    void test_huge_reserve_is_rejected()
    {
        const std::uint64_t impossible_reserve =
            std::numeric_limits<std::uint64_t>::max() / 2;
        const DiskBudgetGuard guard(impossible_reserve);
        const std::filesystem::path here = std::filesystem::current_path();

        bool threw_with_expected_message = false;
        try
        {
            guard.require_capacity(here, 1);
        }
        catch (const std::runtime_error &error)
        {
            const std::string message = error.what();
            threw_with_expected_message =
                message.find("disk reserve exhausted") != std::string::npos;
        }
        require_true(
            threw_with_expected_message,
            "an impossible reserve must raise a disk reserve exhausted error"
        );
    }

    void test_nonexistent_nested_path_resolves_to_existing_ancestor()
    {
        const DiskBudgetGuard guard(0);
        const std::filesystem::path missing =
            std::filesystem::current_path() / "does-not-exist" / "nested" / "dir";
        guard.require_capacity(missing, 1);
    }

    void test_estimate_frame_write_bytes_accounts_for_enrichment_image()
    {
        const std::uint64_t durable_only = estimate_frame_write_bytes(1024, false);
        const std::uint64_t with_enrichment = estimate_frame_write_bytes(1024, true);
        require_true(durable_only > 0, "durable estimate must be positive");
        require_true(
            with_enrichment == durable_only * 2,
            "enrichment image must double the conservative byte estimate"
        );
    }

    void test_estimate_frame_write_bytes_uses_fallback_edge_for_unconstrained_config()
    {
        const std::uint64_t unconstrained = estimate_frame_write_bytes(0, false);
        const std::uint64_t explicit_default = estimate_frame_write_bytes(1024, false);
        require_true(
            unconstrained == explicit_default,
            "a non-positive long edge must fall back to the default estimate edge"
        );
    }

} // namespace

int main()
{
    test_zero_reserve_allows_tiny_write();
    test_huge_reserve_is_rejected();
    test_nonexistent_nested_path_resolves_to_existing_ancestor();
    test_estimate_frame_write_bytes_accounts_for_enrichment_image();
    test_estimate_frame_write_bytes_uses_fallback_edge_for_unconstrained_config();
    std::cout << "test_disk_guard: all tests passed" << std::endl;
    return 0;
}
