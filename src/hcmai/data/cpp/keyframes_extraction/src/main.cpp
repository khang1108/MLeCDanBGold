/**
 * @file main.cpp
 * @brief Provides the native keyframe extractor command-line entry point.
 *
 * This module owns only command-line parsing, exit-code selection, extraction
 * summary formatting, and dispatch to guarded state commands. Source
 * preparation, FFmpeg extraction, and lifecycle mutations remain in library
 * modules so local smoke tests use the same implementation paths.
 */

#include "hcmai/keyframes_extraction/extractor.hpp"
#include "hcmai/keyframes_extraction/state.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

/**
 * @brief Prints the supported command forms to the standard error stream.
 *
 * @return None; writes human-readable usage text for invalid CLI requests.
 */
void print_usage() {
    std::cerr
        << "usage:\n"
        << "  keyframe_extractor --version\n"
        << "  keyframe_extractor extract --manifest <path> --run-root <path> "
           "--config <path> [--video-id <video_id>] "
           "[--source-root <directory>] [--fail-fast]\n"
        << "  keyframe_extractor state mark-enriched --run-root <path> "
           "--video-id <video_id> --artifacts <handoff.json>\n"
        << "  keyframe_extractor state mark-published --run-root <path> "
           "--video-id <video_id> --manifest <manifest.json>\n"
        << "  keyframe_extractor state cleanup --run-root <path> "
           "--video-id <video_id>\n";
}

/**
 * @brief Assigns one non-empty CLI option value while rejecting duplicates.
 *
 * @param destination Optional target that receives the supplied value once.
 * @param option_name Option spelling used in validation diagnostics.
 * @param value Literal command-line value following option_name.
 * @return None; destination contains value after a successful call.
 * @throws std::invalid_argument If destination already has a value or value is blank.
 */
void assign_option(
    std::optional<std::string>& destination,
    std::string_view option_name,
    std::string_view value
) {
    if (destination.has_value()) {
        throw std::invalid_argument("duplicate CLI option: " + std::string(option_name));
    }
    if (value.empty()) {
        throw std::invalid_argument("CLI option value must not be blank: " +
                                    std::string(option_name));
    }
    destination = std::string(value);
}

/**
 * @brief Reads one option value that must immediately follow its option token.
 *
 * @param argc Total number of process command-line arguments.
 * @param argv Process command-line argument array.
 * @param index Index of the current option; advanced to its value on success.
 * @param option_name Option spelling used in diagnostics.
 * @return Literal non-empty value following the current option.
 * @throws std::invalid_argument If option_name has no following argument.
 */
std::string_view next_option_value(
    int argc,
    char* argv[],
    int& index,
    std::string_view option_name
) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            "missing value for CLI option: " + std::string(option_name)
        );
    }
    ++index;
    return argv[index];
}

/**
 * @brief Parses the `extract` command into the library's public request value.
 *
 * @param argc Total number of process command-line arguments.
 * @param argv Process command-line argument array.
 * @return Fully populated ExtractionRequest with all required paths present.
 * @throws std::invalid_argument If options are unknown, duplicated, or incomplete.
 */
hcmai::keyframes_extraction::ExtractionRequest parse_extract_request(
    int argc,
    char* argv[]
) {
    std::optional<std::string> manifest;
    std::optional<std::string> run_root;
    std::optional<std::string> config;
    std::optional<std::string> video_id;
    std::optional<std::string> source_root;
    bool fail_fast = false;

    for (int index = 2; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--fail-fast") {
            if (fail_fast) {
                throw std::invalid_argument("duplicate CLI option: --fail-fast");
            }
            fail_fast = true;
            continue;
        }

        if (option == "--manifest") {
            assign_option(
                manifest,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--run-root") {
            assign_option(
                run_root,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--config") {
            assign_option(
                config,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--video-id") {
            assign_option(
                video_id,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--source-root") {
            assign_option(
                source_root,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else {
            throw std::invalid_argument("unknown CLI option: " + std::string(option));
        }
    }

    if (!manifest.has_value() || !run_root.has_value() || !config.has_value()) {
        throw std::invalid_argument(
            "extract requires --manifest, --run-root, and --config"
        );
    }

    return hcmai::keyframes_extraction::ExtractionRequest{
        std::filesystem::path(manifest.value()),
        std::filesystem::path(run_root.value()),
        std::filesystem::path(config.value()),
        video_id,
        source_root.has_value()
            ? std::optional<std::filesystem::path>(
                std::filesystem::path(source_root.value())
            )
            : std::nullopt,
        fail_fast,
    };
}

/**
 * @brief Writes one machine-readable extraction summary to standard output.
 *
 * @param summary Completed library extraction accounting to serialize.
 * @return None; emits one compact JSON object followed by a newline.
 */
void print_summary(const hcmai::keyframes_extraction::ExtractionSummary& summary) {
    std::cout
        << "{\"completed\":" << summary.completed
        << ",\"failed\":" << summary.failed
        << ",\"skipped\":" << summary.skipped
        << ",\"pending\":" << summary.pending
        << ",\"emitted_frame_count\":" << summary.emitted_frame_count
        << "}\n";
}

/**
 * @brief Executes the parsed extract command and maps its result to exit codes.
 *
 * @param argc Total number of process command-line arguments.
 * @param argv Process command-line argument array.
 * @return Zero when all selected videos complete, two when any video failed,
 *         or one when CLI/config/top-level input is invalid.
 */
int run_extract_command(int argc, char* argv[]) {
    try {
        const hcmai::keyframes_extraction::ExtractionRequest request =
            parse_extract_request(argc, argv);
        const hcmai::keyframes_extraction::ExtractionSummary summary =
            hcmai::keyframes_extraction::extract_manifest(request);
        print_summary(summary);
        return summary.failed == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "keyframe_extractor: " << error.what() << '\n';
        return 1;
    }
}

/**
 * @brief Enumerates the native lifecycle actions exposed by the CLI.
 */
enum class StateCommandAction {
    /** @brief Accept a validated specialist-enrichment handoff. */
    MarkEnriched,
    /** @brief Move an enriched staging bundle into durable publication. */
    MarkPublished,
    /** @brief Remove temporary source and staging artifacts after publication. */
    Cleanup,
};

/**
 * @brief Stores fully parsed command-line inputs for one state action.
 */
struct StateCommandRequest {
    /** @brief Native state mutation selected by the caller. */
    StateCommandAction action;
    /** @brief Root directory owning state, staging, source, and published data. */
    std::filesystem::path run_root;
    /** @brief Canonical video identifier whose lifecycle is being advanced. */
    std::string video_id;
    /** @brief Optional handoff or native manifest path required by the action. */
    std::optional<std::filesystem::path> artifact_path;
};

/**
 * @brief Parses a state action token into its stable CLI enum value.
 *
 * @param action Literal subcommand following the `state` command.
 * @return Matching StateCommandAction value.
 * @throws std::invalid_argument If action is unsupported.
 */
StateCommandAction parse_state_action(std::string_view action) {
    if (action == "mark-enriched") {
        return StateCommandAction::MarkEnriched;
    }
    if (action == "mark-published") {
        return StateCommandAction::MarkPublished;
    }
    if (action == "cleanup") {
        return StateCommandAction::Cleanup;
    }
    throw std::invalid_argument("unknown state action: " + std::string(action));
}

/**
 * @brief Parses one native state subcommand and its action-specific options.
 *
 * @param argc Total number of process command-line arguments.
 * @param argv Process command-line argument array.
 * @return Complete StateCommandRequest with only valid action-specific inputs.
 * @throws std::invalid_argument If an option is unknown, duplicated, or missing.
 */
StateCommandRequest parse_state_request(int argc, char* argv[]) {
    if (argc < 3) {
        throw std::invalid_argument("state requires an action");
    }

    const StateCommandAction action = parse_state_action(argv[2]);
    std::optional<std::string> run_root;
    std::optional<std::string> video_id;
    std::optional<std::string> artifacts;
    std::optional<std::string> manifest;

    for (int index = 3; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--run-root") {
            assign_option(
                run_root,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--video-id") {
            assign_option(
                video_id,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--artifacts") {
            assign_option(
                artifacts,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else if (option == "--manifest") {
            assign_option(
                manifest,
                option,
                next_option_value(argc, argv, index, option)
            );
        } else {
            throw std::invalid_argument("unknown CLI option: " + std::string(option));
        }
    }

    if (!run_root.has_value() || !video_id.has_value()) {
        throw std::invalid_argument(
            "state actions require --run-root and --video-id"
        );
    }

    if (action == StateCommandAction::MarkEnriched) {
        if (!artifacts.has_value() || manifest.has_value()) {
            throw std::invalid_argument(
                "state mark-enriched requires --artifacts and rejects --manifest"
            );
        }
        return StateCommandRequest{
            action,
            std::filesystem::path(run_root.value()),
            video_id.value(),
            std::filesystem::path(artifacts.value()),
        };
    }
    if (action == StateCommandAction::MarkPublished) {
        if (!manifest.has_value() || artifacts.has_value()) {
            throw std::invalid_argument(
                "state mark-published requires --manifest and rejects --artifacts"
            );
        }
        return StateCommandRequest{
            action,
            std::filesystem::path(run_root.value()),
            video_id.value(),
            std::filesystem::path(manifest.value()),
        };
    }
    if (artifacts.has_value() || manifest.has_value()) {
        throw std::invalid_argument("state cleanup does not accept artifact options");
    }
    return StateCommandRequest{
        action,
        std::filesystem::path(run_root.value()),
        video_id.value(),
        std::nullopt,
    };
}

/**
 * @brief Executes one parsed native lifecycle state command.
 *
 * @param argc Total number of process command-line arguments.
 * @param argv Process command-line argument array.
 * @return Zero on a successful or exact idempotent command; one on validation or
 *         filesystem failure.
 */
int run_state_command(int argc, char* argv[]) {
    try {
        const StateCommandRequest request = parse_state_request(argc, argv);
        if (request.action == StateCommandAction::MarkEnriched) {
            hcmai::keyframes_extraction::mark_video_enriched(
                request.run_root,
                request.video_id,
                request.artifact_path.value()
            );
        } else if (request.action == StateCommandAction::MarkPublished) {
            hcmai::keyframes_extraction::mark_video_published(
                request.run_root,
                request.video_id,
                request.artifact_path.value()
            );
        } else {
            hcmai::keyframes_extraction::cleanup_video(
                request.run_root,
                request.video_id
            );
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "keyframe_extractor: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace

/**
 * @brief Parses native extractor commands and dispatches extraction work.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Command-specific status: zero for complete success, two for
 *         per-video failures, and one for invalid or unsupported usage.
 */
int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "hcmai-keyframes-extractor/0.1.0\n";
        return 0;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "extract") {
        return run_extract_command(argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "state") {
        return run_state_command(argc, argv);
    }

    print_usage();
    return 1;
}
