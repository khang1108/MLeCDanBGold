from types import SimpleNamespace

from hcmai.common.schemas import RetrievalCandidate
from hcmai.kis.ranking import KISRankingConfig, shape_kis_candidates


class Data:
    def __init__(self, frames):
        self.frames = frames

    def get_frame(self, frame_id):
        video_id, timestamp_ms = self.frames[frame_id]
        return SimpleNamespace(video_id=video_id, timestamp_ms=timestamp_ms)


def _candidate(frame_id, score):
    return RetrievalCandidate(frame_id=frame_id, final_score=score)


def test_temporal_dedup_keeps_best_and_preserves_suppressed_alternates():
    data = Data({
        "a1": ("video-a", 1_000),
        "a2": ("video-a", 1_400),
        "a3": ("video-a", 4_000),
        "b1": ("video-b", 1_200),
    })
    candidates = [
        _candidate("a1", 1.0),
        _candidate("a2", 0.9),
        _candidate("b1", 0.8),
        _candidate("a3", 0.7),
    ]

    shaped = shape_kis_candidates(
        candidates,
        data,
        KISRankingConfig(temporal_window_ms=500),
    )

    assert [item.frame_id for item in shaped] == ["a1", "b1", "a3"]
    assert shaped[0].metadata["temporal_alternate_frame_ids"] == ["a2"]


def test_diversity_preserves_top_one_and_per_video_order():
    frames = {
        **{f"a{i}": ("video-a", i * 2_000) for i in range(1, 5)},
        "b1": ("video-b", 2_000),
        "c1": ("video-c", 2_000),
    }
    candidates = [
        _candidate("a1", 1.0),
        _candidate("a2", 0.9),
        _candidate("a3", 0.8),
        _candidate("a4", 0.7),
        _candidate("b1", 0.6),
        _candidate("c1", 0.5),
    ]

    shaped = shape_kis_candidates(
        candidates,
        Data(frames),
        KISRankingConfig(
            temporal_window_ms=0,
            early_diversity_depth=5,
            max_per_video_early=2,
        ),
    )

    ids = [item.frame_id for item in shaped]
    assert ids == ["a1", "a2", "b1", "c1", "a3", "a4"]
    assert ids[0] == "a1"
    assert [item for item in ids if item.startswith("a")] == [
        "a1", "a2", "a3", "a4"
    ]


def test_empty_candidates_are_supported():
    assert shape_kis_candidates([], Data({}), KISRankingConfig()) == []
