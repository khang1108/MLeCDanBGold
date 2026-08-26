/**
 * @file config.hpp
 * @brief Declares the native extractor's validated runtime configuration API.
 *
 * This header exposes configuration loading and validation. It does not own
 * JSON parsing internals or apply configuration to extraction work.
 */

#pragma once

#include "hcmai/keyframes_extraction/types.hpp"

#include <filesystem>

namespace hcmai::keyframes_extraction
{

    /**
     * @brief Loads and validates an extraction configuration JSON document.
     *
     * @param path Filesystem path to the JSON configuration document.
     * @return A fully populated, validated ExtractionConfig value.
     * @throws std::runtime_error If the file cannot be read.
     * @throws std::invalid_argument If JSON or configuration fields are invalid.
     */
    ExtractionConfig read_extraction_config(const std::filesystem::path &path);

    /**
     * @brief Enforces required extractor configuration invariants.
     *
     * @param config The configuration value to validate.
     * @return None; throws when an invariant is not satisfied.
     * @throws std::invalid_argument If any configuration value is invalid.
     */
    void validate_extraction_config(const ExtractionConfig &config);

} // namespace hcmai::keyframes_extraction
