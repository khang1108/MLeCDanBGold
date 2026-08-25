/**
 * @file test_state.cpp
 * @brief Verifies atomic native per-video state and guarded lifecycle changes.
 *
 * The fixture uses one synthetic video state so failed transitions can prove
 * that the persisted checkpoint remains unchanged.
 */

#include "hcmai/keyframes_extraction/state.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <string>

/**
 * @brief Exercises state serialization, atomic replacement, and provenance guards.
 *
 * @return Zero when the state contract holds; throws otherwise.
 */
int main() {
    using hcmai::keyframes_extraction::StateIdentity;
    using hcmai::keyframes_extraction::VideoStatus;
    using hcmai::keyframes_extraction::VideoState;
    using hcmai::keyframes_extraction::make_pending_state;
    using hcmai::keyframes_extraction::read_state;
    using hcmai::keyframes_extraction::save_state_atomic;
    using hcmai::keyframes_extraction::transition_state;
    using namespace hcmai::keyframes_extraction::test_support;

    const std::filesystem::path root = make_temp_directory("state");
    const std::filesystem::path state_path =
        root / "state" / "L01_V001.json";
    const StateIdentity identity{
        "run-1",
        "L01_V001",
        "hcmai-keyframes-extractor/0.1.0",
        "hash-1",
    };

    VideoState state = make_pending_state(
        identity,
        "https://youtube.com/watch?v=a"
    );
    save_state_atomic(state_path, state);

    require_true(hcmai::keyframes_extraction::test_support::exists(state_path),
                 "atomic state write must publish final path");
    require_true(!hcmai::keyframes_extraction::test_support::exists(
                     state_path.string() + ".tmp"
                 ),
                 "successful state write must not retain a temporary file");

    const VideoState pending = read_state(state_path);
    require_true(pending.status == VideoStatus::Pending,
                 "new state must start as pending");
    require_true(pending.run_id == identity.run_id,
                 "state must retain its run identity");
    require_true(pending.updated_at == pending.started_at,
                 "new state timestamps must begin at the same instant");

    state = transition_state(
        state_path,
        identity,
        VideoStatus::Pending,
        VideoStatus::Downloading
    );
    state = transition_state(
        state_path,
        identity,
        VideoStatus::Downloading,
        VideoStatus::Extracting
    );
    require_true(state.status == VideoStatus::Extracting,
                 "forward lifecycle transitions must persist");

    require_throws([&] {
        transition_state(
            state_path,
            identity,
            VideoStatus::Extracting,
            VideoStatus::Cleaned
        );
    });
    require_true(read_state(state_path).status == VideoStatus::Extracting,
                 "rejected lifecycle transition must not mutate state");

    StateIdentity changed_identity = identity;
    changed_identity.config_hash = "hash-2";
    require_throws([&] {
        transition_state(
            state_path,
            changed_identity,
            VideoStatus::Extracting,
            VideoStatus::Extracted
        );
    });
    require_true(read_state(state_path).status == VideoStatus::Extracting,
                 "provenance mismatch must not mutate state");

    const std::string oversized_error(
        hcmai::keyframes_extraction::kMaxStoredStateErrorBytes + 1U,
        'x'
    );
    state = transition_state(
        state_path,
        identity,
        VideoStatus::Extracting,
        VideoStatus::Failed,
        oversized_error
    );
    require_true(state.status == VideoStatus::Failed,
                 "active lifecycle state must be able to fail");
    require_true(
        state.error.size() ==
            hcmai::keyframes_extraction::kMaxStoredStateErrorBytes,
        "failed state diagnostics must be bounded"
    );

    state = transition_state(
        state_path,
        identity,
        VideoStatus::Failed,
        VideoStatus::Downloading
    );
    require_true(state.status == VideoStatus::Downloading,
                 "failed video must be allowed to restart source preparation");
    require_true(state.error.empty(),
                 "retry state must clear the previous failure diagnostic");

    return finish_test();
}
