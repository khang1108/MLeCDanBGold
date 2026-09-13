"""Tests for reusable temporal scoring and selected-video decoding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus import Frame
from hcmai.orchestration.workflows import temporal_search
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import DPPath


def _video() -> VideoEventScores:
    """Build one two-event video with canonical frame metadata."""

    return VideoEventScores(
        video_id="v1",
        frame_ids=np.asarray(["f0", "f1", "f2"], dtype=object),
        frame_idx=np.asarray([10, 20, 30]),
        timestamps_ms=np.asarray([1_000, 2_000, 3_000]),
        scores=np.asarray([[0.9, 0.2, 0.1], [0.1, 0.3, 0.8]]),
    )


class FakeCorpus:
    """Provide the canonical records required by score and path validation."""

    frames = {
        "f0": Frame("f0", "v1", 10, 1_000, "f0.jpg"),
        "f1": Frame("f1", "v1", 20, 2_000, "f1.jpg"),
        "f2": Frame("f2", "v1", 30, 3_000, "f2.jpg"),
    }

    def frame(self, frame_id: str) -> Frame:
        """Return the canonical record for one score column."""

        return self.frames[frame_id]


def _service(evidence: object) -> TemporalSearchService:
    """Build a service with deterministic DP settings."""

    return TemporalSearchService(FakeCorpus(), evidence, AlignmentConfig(lambda_gap=0.0))


def test_score_videos_normalizes_and_preserves_modern_source_inputs() -> None:
    """Pass normalized parallel event sequences and source choices unchanged."""

    scorer = Mock(return_value=[])
    service = object.__new__(TemporalSearchService)
    service.max_temporal_event_count = 8
    service.config = SimpleNamespace(chunk_size=64)
    service.evidence = SimpleNamespace(score_events=scorer)
    service._validate_video_scores = Mock()

    videos, elapsed = service.score_videos(
        [" mở tủ ", "đặt   cốc"],
        retrieval_events=[" opens cabinet", "places   cup "],
        caption_events=["mở tủ", " đặt cốc "],
        use_dense=True,
        use_bm25=True,
    )

    scorer.assert_called_once_with(
        ("mở tủ", "đặt cốc"),
        ("opens cabinet", "places cup"),
        caption_events=("mở tủ", "đặt cốc"),
        use_dense=True,
        use_bm25=True,
    )
    assert videos == ()
    assert elapsed >= 0


def test_score_videos_forwards_bm25_only_source_flags() -> None:
    """Preserve disabled Dense routing for valid BM25-only scoring."""

    scorer = Mock(return_value=[])
    service = object.__new__(TemporalSearchService)
    service.max_temporal_event_count = 8
    service.config = SimpleNamespace(chunk_size=64)
    service.evidence = SimpleNamespace(score_events=scorer)
    service._validate_video_scores = Mock()

    service.score_videos(
        ["mở tủ"],
        caption_events=["mở tủ"],
        use_dense=False,
        use_bm25=True,
    )

    scorer.assert_called_once_with(
        ("mở tủ",),
        ("mở tủ",),
        caption_events=("mở tủ",),
        use_dense=False,
        use_bm25=True,
    )


def test_score_videos_timing_excludes_post_score_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop retrieval timing before canonical score validation begins."""

    scored_video = _video()
    scorer = Mock(return_value=[scored_video])
    service = object.__new__(TemporalSearchService)
    service.max_temporal_event_count = 8
    service.config = SimpleNamespace(chunk_size=64)
    service.evidence = SimpleNamespace(score_events=scorer)
    validation_clock_reads: list[float] = []

    def validate(event_count: int, video: VideoEventScores) -> None:
        """Read the patched clock only after score timing has ended."""

        assert event_count == 2
        assert video is scored_video
        validation_clock_reads.append(temporal_search.perf_counter())

    service._validate_video_scores = validate
    monkeypatch.setattr(
        temporal_search,
        "perf_counter",
        Mock(side_effect=[10.0, 10.25, 100.0]),
    )

    _, elapsed = service.score_videos(["first", "second"])

    assert elapsed == pytest.approx(250.0)
    assert validation_clock_reads == [100.0]


@pytest.mark.parametrize(
    ("original", "retrieval", "captions", "use_dense", "use_bm25", "message"),
    [
        (["one", "two"], ["one"], None, True, False, "retrieval_events"),
        (["one", "two"], None, ["one"], True, False, "caption_events"),
        (["one"], None, None, False, False, "at least one"),
        ([], None, None, True, False, "empty"),
        (["one", "two"], None, None, True, False, "at most 1"),
    ],
)
def test_score_videos_validates_before_scorer_call(
    original: list[str],
    retrieval: list[str] | None,
    captions: list[str] | None,
    use_dense: bool,
    use_bm25: bool,
    message: str,
) -> None:
    """Reject invalid parallel source inputs before model work begins."""

    scorer = Mock(return_value=[])
    service = object.__new__(TemporalSearchService)
    service.max_temporal_event_count = 1 if message == "at most 1" else 8
    service.config = SimpleNamespace(chunk_size=64)
    service.evidence = SimpleNamespace(score_events=scorer)
    service._validate_video_scores = Mock()

    with pytest.raises(ValueError, match=message):
        service.score_videos(
            original,
            retrieval_events=retrieval,
            caption_events=captions,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )

    scorer.assert_not_called()


def test_score_videos_uses_legacy_scorer_with_configured_chunk_size() -> None:
    """Keep the compatibility call for legacy temporal evidence scorers."""

    legacy = SimpleNamespace(score_event_videos=Mock(return_value=[]))
    service = object.__new__(TemporalSearchService)
    service.max_temporal_event_count = 8
    service.config = SimpleNamespace(chunk_size=64)
    service.evidence = legacy
    service._validate_video_scores = Mock()

    videos, _ = service.score_videos([" one "])

    legacy.score_event_videos.assert_called_once_with(("one",), chunk_size=64)
    assert videos == ()


def test_validate_video_scores_rejects_canonical_timestamp_drift() -> None:
    """Reject a score timestamp that differs from the corpus record."""

    invalid = VideoEventScores(
        video_id="v1",
        frame_ids=np.asarray(["f0"]),
        frame_idx=np.asarray([10]),
        timestamps_ms=np.asarray([1_001]),
        scores=np.asarray([[0.7]]),
    )

    with pytest.raises(ValueError, match="timestamp"):
        _service(SimpleNamespace())._validate_video_scores(1, invalid)


def test_decode_video_materializes_canonical_identity() -> None:
    """Decode an allowed path using canonical rather than reconstructed times."""

    paths = _service(SimpleNamespace()).decode_video(
        _video(),
        allowed=np.asarray([[True, False, False], [False, False, True]]),
    )

    assert paths[0].frame_ids == ("f0", "f2")
    assert paths[0].frame_idxs == (10, 30)
    assert paths[0].timestamps_ms == (1_000, 3_000)


def test_decode_video_forwards_baseline_config_and_materializes_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate one masked decode with every baseline setting intact."""

    video = _video()
    allowed = np.asarray([[True, False, False], [False, False, True]])
    row = DPPath("v1", 1.2, (10, 30), ("f0", "f2"))
    expected = object()
    decode = Mock(return_value=[row])
    materialize = Mock(return_value=expected)
    service = object.__new__(TemporalSearchService)
    service.config = AlignmentConfig(
        lambda_gap=0.25,
        event_power=0.5,
        cluster_delta=0.75,
        path_min_separation_ms=42,
    )
    service._materialize_aligned_path = materialize
    monkeypatch.setattr(temporal_search, "align_video", decode)

    paths = service.decode_video(video, allowed=allowed)

    assert paths == (expected,)
    assert decode.call_args.args == (video,)
    assert {
        name: value
        for name, value in decode.call_args.kwargs.items()
        if name != "allowed"
    } == {
        "lambda_gap": 0.25,
        "paths": 1,
        "event_power": 0.5,
        "cluster_delta": 0.75,
        "min_separation_ms": 42,
    }
    assert decode.call_args.kwargs["allowed"] is allowed
    assert materialize.call_args_list == [call(row, video)]


def test_search_keeps_deterministic_baseline_paths_after_scoring_split() -> None:
    """Keep the established ranking and canonical path output unchanged."""

    legacy = SimpleNamespace(score_event_videos=Mock(return_value=[_video()]))

    result = _service(legacy).search(["first", "second"], top_k=2)

    assert [
        (path.video_id, path.frame_ids, path.frame_idxs, path.timestamps_ms)
        for path in result.paths
    ] == [
        ("v1", ("f0", "f2"), (10, 30), (1_000, 3_000)),
        ("v1", ("f0", "f1"), (10, 20), (1_000, 2_000)),
    ]


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_nonpositive_top_k_before_scoring(top_k: int) -> None:
    """Retain search's boundary validation before any retrieval work."""

    service = _service(SimpleNamespace())
    service.score_videos = Mock()

    with pytest.raises(ValueError, match="top_k"):
        service.search(["event"], top_k=top_k)

    service.score_videos.assert_not_called()
