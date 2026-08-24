#include "hcmai/keyframes_extraction/jsonl.hpp"

#include <json-c/json.h>
#include <json-c/json_tokener.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace hcmai::keyframes_extraction {
namespace {

class JsonObject {
public:
    explicit JsonObject(json_object* value) : value_(value) {}

    ~JsonObject() {
        if (value_ != nullptr) {
            json_object_put(value_);
        }
    }

    JsonObject(const JsonObject&) = delete;
    JsonObject& operator=(const JsonObject&) = delete;

    json_object* get() const {
        return value_;
    }

private:
    json_object* value_;
};

std::string context(
    const std::filesystem::path& path,
    std::size_t line_number
) {
    return path.string() + ":" + std::to_string(line_number);
}

JsonObject parse_object(
    const std::string& line,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    json_tokener_error error = json_tokener_success;
    json_object* parsed = json_tokener_parse_verbose(line.c_str(), &error);
    if (parsed == nullptr || error != json_tokener_success) {
        throw std::invalid_argument(
            "invalid JSON at " + context(path, line_number) + ": " +
            json_tokener_error_desc(error)
        );
    }
    if (!json_object_is_type(parsed, json_type_object)) {
        json_object_put(parsed);
        throw std::invalid_argument(
            "JSONL row must be an object at " + context(path, line_number)
        );
    }
    return JsonObject(parsed);
}

json_object* required_member(
    json_object* object,
    const char* key,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    json_object* value = nullptr;
    if (!json_object_object_get_ex(object, key, &value) || value == nullptr) {
        throw std::invalid_argument(
            "missing field '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    return value;
}

bool is_blank(std::string_view value) {
    return std::all_of(
        value.begin(),
        value.end(),
        [](unsigned char character) {
            return std::isspace(character) != 0;
        }
    );
}

std::string required_string(
    json_object* object,
    const char* key,
    const std::filesystem::path& path,
    std::size_t line_number,
    bool allow_blank = false
) {
    json_object* value = required_member(object, key, path, line_number);
    if (!json_object_is_type(value, json_type_string)) {
        throw std::invalid_argument(
            "field must be a string '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    const char* raw = json_object_get_string(value);
    const std::string result = raw == nullptr ? std::string() : std::string(raw);
    if (!allow_blank && (result.empty() || is_blank(result))) {
        throw std::invalid_argument(
            "field must not be blank '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    return result;
}

std::int64_t required_integer(
    json_object* object,
    const char* key,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    json_object* value = required_member(object, key, path, line_number);
    if (!json_object_is_type(value, json_type_int)) {
        throw std::invalid_argument(
            "field must be an integer '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    return json_object_get_int64(value);
}

double required_number(
    json_object* object,
    const char* key,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    json_object* value = required_member(object, key, path, line_number);
    if (!json_object_is_type(value, json_type_int) &&
        !json_object_is_type(value, json_type_double)) {
        throw std::invalid_argument(
            "field must be numeric '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    const double result = json_object_get_double(value);
    if (!std::isfinite(result)) {
        throw std::invalid_argument(
            "field must be finite '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    return result;
}

std::uint64_t non_negative_u64(
    std::int64_t value,
    const char* key,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    if (value < 0) {
        throw std::invalid_argument(
            "field must be non-negative '" + std::string(key) + "' at " +
            context(path, line_number)
        );
    }
    return static_cast<std::uint64_t>(value);
}

bool valid_video_id(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(
        value.begin(),
        value.end(),
        [](unsigned char character) {
            return std::isalnum(character) != 0 ||
                character == '_' || character == '-' || character == '.';
        }
    );
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open JSONL file: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

VideoInput parse_video_input(
    const std::string& line,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    const JsonObject root = parse_object(line, path, line_number);
    const std::string video_id = required_string(
        root.get(), "video_id", path, line_number
    );
    if (!valid_video_id(video_id)) {
        throw std::invalid_argument(
            "video_id contains unsafe characters at " +
            context(path, line_number)
        );
    }

    const std::string watch_url = required_string(
        root.get(), "watch_url", path, line_number
    );
    if (watch_url.rfind("http://", 0) != 0 &&
        watch_url.rfind("https://", 0) != 0) {
        throw std::invalid_argument(
            "watch_url must use http or https at " +
            context(path, line_number)
        );
    }

    const std::int64_t metadata_length_s = required_integer(
        root.get(), "metadata_length_s", path, line_number
    );
    if (metadata_length_s < 0) {
        throw std::invalid_argument(
            "metadata_length_s must be non-negative at " +
            context(path, line_number)
        );
    }
    return VideoInput{video_id, watch_url, metadata_length_s};
}

NativeFrameRow parse_frame_row(
    const std::string& line,
    const std::filesystem::path& path,
    std::size_t line_number
) {
    const JsonObject root = parse_object(line, path, line_number);
    const std::string frame_id = required_string(
        root.get(), "frame_id", path, line_number
    );
    const std::string video_id = required_string(
        root.get(), "video_id", path, line_number
    );
    if (!valid_video_id(video_id)) {
        throw std::invalid_argument(
            "frame video_id contains unsafe characters at " +
            context(path, line_number)
        );
    }

    const auto sample_index = non_negative_u64(
        required_integer(root.get(), "sample_index", path, line_number),
        "sample_index",
        path,
        line_number
    );
    const auto target_timestamp_ms = non_negative_u64(
        required_integer(
            root.get(),
            "target_timestamp_ms",
            path,
            line_number
        ),
        "target_timestamp_ms",
        path,
        line_number
    );
    const auto timestamp_ms = non_negative_u64(
        required_integer(root.get(), "timestamp_ms", path, line_number),
        "timestamp_ms",
        path,
        line_number
    );
    const auto frame_idx = non_negative_u64(
        required_integer(root.get(), "frame_idx", path, line_number),
        "frame_idx",
        path,
        line_number
    );
    const double avg_fps = required_number(
        root.get(), "avg_fps", path, line_number
    );
    if (avg_fps <= 0.0) {
        throw std::invalid_argument(
            "avg_fps must be positive at " + context(path, line_number)
        );
    }

    const auto avg_fps_num = non_negative_u64(
        required_integer(root.get(), "avg_fps_num", path, line_number),
        "avg_fps_num",
        path,
        line_number
    );
    const auto avg_fps_den = non_negative_u64(
        required_integer(root.get(), "avg_fps_den", path, line_number),
        "avg_fps_den",
        path,
        line_number
    );
    const auto time_base_num = non_negative_u64(
        required_integer(root.get(), "time_base_num", path, line_number),
        "time_base_num",
        path,
        line_number
    );
    const auto time_base_den = non_negative_u64(
        required_integer(root.get(), "time_base_den", path, line_number),
        "time_base_den",
        path,
        line_number
    );
    if (avg_fps_num == 0 || avg_fps_den == 0 ||
        time_base_num == 0 || time_base_den == 0) {
        throw std::invalid_argument(
            "rational metadata must be positive at " +
            context(path, line_number)
        );
    }

    const std::int64_t pts = required_integer(
        root.get(), "pts", path, line_number
    );
    const auto width = required_integer(
        root.get(), "width", path, line_number
    );
    const auto height = required_integer(
        root.get(), "height", path, line_number
    );
    if (width <= 0 || height <= 0 ||
        width > std::numeric_limits<int>::max() ||
        height > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "image dimensions must be positive and fit int at " +
            context(path, line_number)
        );
    }

    const std::string image_path = required_string(
        root.get(), "image_path", path, line_number
    );
    const std::string enrichment_image_path = required_string(
        root.get(),
        "enrichment_image_path",
        path,
        line_number,
        true
    );
    const auto image_size_bytes = non_negative_u64(
        required_integer(root.get(), "image_size_bytes", path, line_number),
        "image_size_bytes",
        path,
        line_number
    );
    const auto enrichment_image_size_bytes = non_negative_u64(
        required_integer(
            root.get(),
            "enrichment_image_size_bytes",
            path,
            line_number
        ),
        "enrichment_image_size_bytes",
        path,
        line_number
    );
    if (image_size_bytes == 0 ||
        (!enrichment_image_path.empty() && enrichment_image_size_bytes == 0)) {
        throw std::invalid_argument(
            "encoded image sizes must be positive at " +
            context(path, line_number)
        );
    }

    return NativeFrameRow{
        frame_id,
        video_id,
        sample_index,
        static_cast<std::int64_t>(target_timestamp_ms),
        static_cast<std::int64_t>(timestamp_ms),
        static_cast<std::int64_t>(frame_idx),
        avg_fps,
        RationalValue{
            static_cast<std::int64_t>(avg_fps_num),
            static_cast<std::int64_t>(avg_fps_den),
        },
        pts,
        RationalValue{
            static_cast<std::int64_t>(time_base_num),
            static_cast<std::int64_t>(time_base_den),
        },
        static_cast<int>(width),
        static_cast<int>(height),
        image_path,
        enrichment_image_path,
        image_size_bytes,
        enrichment_image_size_bytes,
    };
}

void add_string(json_object* object, const char* key, const std::string& value) {
    json_object_object_add(
        object,
        key,
        json_object_new_string(value.c_str())
    );
}

void add_integer(json_object* object, const char* key, std::int64_t value) {
    json_object_object_add(object, key, json_object_new_int64(value));
}

void add_unsigned(json_object* object, const char* key, std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()
        )) {
        throw std::overflow_error(
            "JSON integer exceeds json-c signed integer range"
        );
    }
    add_integer(object, key, static_cast<std::int64_t>(value));
}

void validate_frame_row_for_write(const NativeFrameRow& row) {
    if (row.frame_id.empty() || row.video_id.empty() ||
        !valid_video_id(row.video_id)) {
        throw std::invalid_argument("native frame identity must be valid");
    }
    if (row.target_timestamp_ms < 0 || row.timestamp_ms < 0 ||
        row.frame_idx < 0) {
        throw std::invalid_argument(
            "native frame timestamps must be non-negative"
        );
    }
    if (!std::isfinite(row.avg_fps) || row.avg_fps <= 0.0 ||
        row.avg_fps_rational.numerator <= 0 ||
        row.avg_fps_rational.denominator <= 0 ||
        row.time_base.numerator <= 0 || row.time_base.denominator <= 0) {
        throw std::invalid_argument(
            "native frame rational metadata is invalid"
        );
    }
    if (row.width <= 0 || row.height <= 0 || row.image_path.empty() ||
        row.image_size_bytes == 0) {
        throw std::invalid_argument("native frame image metadata is invalid");
    }
    if (row.enrichment_image_path.empty() &&
        row.enrichment_image_size_bytes != 0) {
        throw std::invalid_argument(
            "empty enrichment image path requires zero byte size"
        );
    }
    if (!row.enrichment_image_path.empty() &&
        row.enrichment_image_size_bytes == 0) {
        throw std::invalid_argument(
            "enrichment image path requires positive byte size"
        );
    }
}

}  // namespace

std::vector<VideoInput> read_video_manifest(
    const std::filesystem::path& path
) {
    const auto lines = read_lines(path);
    std::vector<VideoInput> rows;
    std::unordered_set<std::string> video_ids;
    std::unordered_set<std::string> watch_urls;
    rows.reserve(lines.size());

    for (std::size_t index = 0; index < lines.size(); ++index) {
        const std::string& line = lines[index];
        if (line.empty() || is_blank(line)) {
            throw std::invalid_argument(
                "blank JSONL row at " + context(path, index + 1)
            );
        }
        VideoInput row = parse_video_input(line, path, index + 1);
        if (!video_ids.insert(row.video_id).second) {
            throw std::invalid_argument(
                "duplicate video_id at " + context(path, index + 1)
            );
        }
        if (!watch_urls.insert(row.watch_url).second) {
            throw std::invalid_argument(
                "duplicate watch_url at " + context(path, index + 1)
            );
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        throw std::invalid_argument("JSONL manifest must contain at least one row: " + path.string());
    }
    return rows;
}

std::vector<NativeFrameRow> read_frame_jsonl(
    const std::filesystem::path& path
) {
    const auto lines = read_lines(path);
    std::vector<NativeFrameRow> rows;
    std::unordered_set<std::string> frame_ids;
    std::unordered_set<std::uint64_t> sample_indexes;
    rows.reserve(lines.size());

    for (std::size_t index = 0; index < lines.size(); ++index) {
        const std::string& line = lines[index];
        if (line.empty() || is_blank(line)) {
            throw std::invalid_argument(
                "blank JSONL row at " + context(path, index + 1)
            );
        }
        NativeFrameRow row = parse_frame_row(line, path, index + 1);
        if (!frame_ids.insert(row.frame_id).second) {
            throw std::invalid_argument(
                "duplicate frame_id at " + context(path, index + 1)
            );
        }
        if (!sample_indexes.insert(row.sample_index).second) {
            throw std::invalid_argument(
                "duplicate sample_index at " + context(path, index + 1)
            );
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void write_frame_row(std::ostream& output, const NativeFrameRow& row) {
    validate_frame_row_for_write(row);
    JsonObject object(json_object_new_object());
    if (object.get() == nullptr) {
        throw std::runtime_error("unable to allocate JSON object");
    }

    add_string(object.get(), "frame_id", row.frame_id);
    add_string(object.get(), "video_id", row.video_id);
    add_unsigned(object.get(), "sample_index", row.sample_index);
    add_integer(object.get(), "target_timestamp_ms", row.target_timestamp_ms);
    add_integer(object.get(), "timestamp_ms", row.timestamp_ms);
    add_integer(object.get(), "frame_idx", row.frame_idx);
    json_object_object_add(
        object.get(),
        "avg_fps",
        json_object_new_double(row.avg_fps)
    );
    add_integer(object.get(), "avg_fps_num", row.avg_fps_rational.numerator);
    add_integer(object.get(), "avg_fps_den", row.avg_fps_rational.denominator);
    add_integer(object.get(), "pts", row.pts);
    add_integer(object.get(), "time_base_num", row.time_base.numerator);
    add_integer(object.get(), "time_base_den", row.time_base.denominator);
    add_integer(object.get(), "width", row.width);
    add_integer(object.get(), "height", row.height);
    add_string(object.get(), "image_path", row.image_path);
    add_string(object.get(), "enrichment_image_path", row.enrichment_image_path);
    add_unsigned(object.get(), "image_size_bytes", row.image_size_bytes);
    add_unsigned(
        object.get(),
        "enrichment_image_size_bytes",
        row.enrichment_image_size_bytes
    );

    output << json_object_to_json_string_ext(
               object.get(),
               JSON_C_TO_STRING_PLAIN
           )
           << '\n';
    if (!output) {
        throw std::runtime_error("unable to write native frame JSONL row");
    }
}

void write_frame_jsonl(
    const std::filesystem::path& path,
    const std::vector<NativeFrameRow>& rows
) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("unable to open native frame JSONL: " + path.string());
    }
    for (const NativeFrameRow& row : rows) {
        write_frame_row(output, row);
    }
}

}  // namespace hcmai::keyframes_extraction
