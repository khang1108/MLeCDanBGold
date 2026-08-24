#pragma once

#include "hcmai/keyframes_extraction/types.hpp"

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace hcmai::keyframes_extraction {

std::vector<VideoInput> read_video_manifest(
    const std::filesystem::path& path
);

std::vector<NativeFrameRow> read_frame_jsonl(
    const std::filesystem::path& path
);

void write_frame_row(
    std::ostream& output,
    const NativeFrameRow& row
);

void write_frame_jsonl(
    const std::filesystem::path& path,
    const std::vector<NativeFrameRow>& rows
);

}  // namespace hcmai::keyframes_extraction
