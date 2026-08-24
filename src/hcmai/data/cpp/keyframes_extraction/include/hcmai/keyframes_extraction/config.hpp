#pragma once

#include "hcmai/keyframes_extraction/types.hpp"

#include <filesystem>

namespace hcmai::keyframes_extraction {

ExtractionConfig read_extraction_config(const std::filesystem::path& path);

void validate_extraction_config(const ExtractionConfig& config);

}  // namespace hcmai::keyframes_extraction
