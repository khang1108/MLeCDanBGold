"""Regression tests for stateful temporal exploration updates."""

import unittest

import numpy as np

from hcmai.orchestration.workflows.temporal_exploration import (
    QueryBinding,
    TemporalExploration,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class _TemporalSearch:
    """Minimal selected-video scorer and decoder fake."""

    def __init__(self) -> None:
        self.decode_calls = 0

    def score_videos(self, *_: object, **__: object) -> tuple[tuple[VideoEventScores, ...], float]:
        """Return one complete canonical score matrix."""
        return (
            (
                VideoEventScores(
                    video_id="video-1",
                    frame_ids=np.array(["frame-1", "frame-2"]),
                    frame_idx=np.array([100, 200]),
                    timestamps_ms=np.array([0, 100], dtype=np.int64),
                    scores=np.array([[0.5, 0.6]], dtype=np.float32),
                ),
            ),
            0.0,
        )

    def snapshot_decoder_config(self) -> None:
        """Use the fake decoder's default configuration."""
        return None

    def decode_video(self, *_: object, **__: object) -> tuple[()]:
        """Record a decode attempt without constructing a path."""
        self.decode_calls += 1
        return ()


class TemporalExplorationTest(unittest.TestCase):
    """Protect state transitions while simplifying evaluator internals."""

    def test_window_update_re_evaluates_before_publishing_a_new_revision(self) -> None:
        temporal = _TemporalSearch()
        exploration = TemporalExploration(temporal)  # type: ignore[arg-type]
        binding = QueryBinding(
            query="A person walks",
            event_version="events-v1",
            events=("A person walks",),
            retrieval_events=("A person walks",),
            caption_events=None,
            use_dense=True,
            use_bm25=False,
            scoring_revision="scores-v1",
        )

        opened = exploration.open(binding, "video-1", (0, 100))
        updated = exploration.apply(
            expected_revision=opened.revision,
            event_version=binding.event_version,
            scoring_revision=binding.scoring_revision,
            action="window",
            interval=(0, 99),
        )

        self.assertEqual(temporal.decode_calls, 2)
        self.assertEqual(opened.revision, 1)
        self.assertEqual(opened.conditions.window, (0, 100))
        self.assertEqual(updated.revision, 2)
        self.assertEqual(updated.conditions.window, (0, 99))
        self.assertEqual(exploration.current(), updated)


if __name__ == "__main__":
    unittest.main()
