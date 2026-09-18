from hcmai.common.config import AvsConfig
from hcmai.orchestration.workflows.avs import AvsCoverageCandidate, AvsCoverageSelector


def c(frame_id: str, video_id: str, ts: int, rank: int, score: float) -> AvsCoverageCandidate:
    return AvsCoverageCandidate(
        frame_id=frame_id,
        video_id=video_id,
        timestamp_ms=ts,
        retrieval_rank=rank,
        retrieval_score=score,
    )


def selector(window_ms: int = 3000) -> AvsCoverageSelector:
    return AvsCoverageSelector(AvsConfig(temporal_dedup_window_ms=window_ms))


def test_config_requires_pool_larger_than_visible_page():
    try:
        AvsConfig(candidate_pool_size=80, maximum_page_size=80)
    except ValueError:
        pass
    else:
        raise AssertionError("candidate pool must be larger than maximum visible page")


def test_same_video_nearby_candidates_keep_strongest_retrieval_rank():
    ordered = selector().order([
        c("v1-a", "V1", 10_000, 1, .99),
        c("v1-b", "V1", 11_000, 2, .98),
        c("v2-a", "V2", 20_000, 3, .97),
    ])
    assert [item.frame_id for item in ordered] == ["v1-a", "v2-a"]


def test_same_video_candidates_outside_window_remain_eligible():
    ordered = selector(window_ms=2000).order([
        c("v1-a", "V1", 10_000, 1, .99),
        c("v1-b", "V1", 13_000, 2, .98),
    ])
    assert [item.frame_id for item in ordered] == ["v1-a", "v1-b"]


def test_video_first_pass_exposes_other_videos_before_second_v1_candidate():
    ordered = selector(window_ms=0).order([
        c("v1-a", "V1", 10_000, 1, .99),
        c("v1-b", "V1", 20_000, 2, .98),
        c("v1-c", "V1", 30_000, 3, .97),
        c("v2-a", "V2", 40_000, 4, .96),
        c("v3-a", "V3", 50_000, 5, .95),
    ])
    assert [item.frame_id for item in ordered] == [
        "v1-a", "v2-a", "v3-a", "v1-b", "v1-c",
    ]


def test_selector_is_deterministic_and_preserves_all_non_suppressed_candidates():
    values = [
        c("v2-b", "V2", 40_000, 4, .70),
        c("v1-a", "V1", 10_000, 1, .99),
        c("v2-a", "V2", 20_000, 2, .90),
        c("v1-b", "V1", 30_000, 3, .80),
    ]
    first = selector(window_ms=0).order(values)
    second = selector(window_ms=0).order(list(reversed(values)))
    assert first == second
    assert {item.frame_id for item in first} == {item.frame_id for item in values}
