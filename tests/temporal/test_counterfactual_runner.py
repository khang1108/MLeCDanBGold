"""Unit tests for counterfactual experiment runner (SP-14)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.diagnostic import DiagnosticDataset, DiagnosticEvent, DiagnosticQuery
from scripts.evaluation.run_counterfactuals import (
    evaluate_reverse_counterfactuals,
    evaluate_shuffled_counterfactuals,
    save_counterfactual_result,
)


@pytest.fixture
def synthetic_diagnostic_setup():
    """Create synthetic diagnostic queries, video scores, and frame embeddings."""
    q1 = DiagnosticQuery(
        query_id="diag_001",
        video_id="video_001",
        query="person sitting then person standing",
        category="motion_direction",
        events=(
            DiagnosticEvent(id="e1", text="person sitting", target_frame_idxs=(2,)),
            DiagnosticEvent(id="e2", text="person standing", target_frame_idxs=(8,)),
        ),
    )
    dataset = DiagnosticDataset(queries=[q1])

    # 10 frames
    n_frames = 10
    frame_ids = [f"f_{i:03d}" for i in range(n_frames)]
    frame_idx = np.arange(n_frames, dtype=int)
    timestamps = (frame_idx * 1000).astype(float)

    # Scores: event 0 prefers frame 2, event 1 prefers frame 8
    scores = np.zeros((2, n_frames), dtype=np.float32)
    scores[0, 2] = 0.9
    scores[1, 8] = 0.9

    video_scores = {
        "video_001": VideoEventScores(
            video_id="video_001",
            frame_ids=frame_ids,
            frame_idx=frame_idx,
            timestamps_ms=timestamps,
            scores=scores,
        )
    }

    # Embeddings: 4D vectors where direction along dim 0 indicates motion
    embeddings_map = {
        "video_001": np.array(
            [[float(i), 0.0, 0.0, 0.0] for i in range(n_frames)],
            dtype=np.float32,
        )
    }

    # Normalize embeddings
    norms = np.linalg.norm(embeddings_map["video_001"], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_map["video_001"] = embeddings_map["video_001"] / norms

    def frame_accessor(vid: str, idxs: np.ndarray) -> np.ndarray:
        return embeddings_map[vid][idxs]

    return dataset, video_scores, frame_accessor


def test_evaluate_reverse_counterfactuals(synthetic_diagnostic_setup):
    dataset, video_scores, frame_accessor = synthetic_diagnostic_setup

    with patch("scripts.evaluation.run_counterfactuals.encode_query_events") as mock_encode:
        # Mock query event embeddings (2 events, 4D)
        # Event 0: [0, 1, 0, 0], Event 1: [1, 0, 0, 0]
        # Transition dQ = [1, -1, 0, 0]
        mock_encode.side_effect = lambda texts: (
            np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            if texts[0] == "person sitting"
            else np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        )

        res = evaluate_reverse_counterfactuals(
            dataset=dataset,
            video_scores=video_scores,
            frame_embeddings=frame_accessor,
            candidate_k=5,
            transition_weight=0.5,
        )

        assert res["experiment"] == "reverse_counterfactual"
        assert res["num_queries"] == 1
        assert "reverse_accuracy" in res
        assert "mean_margin" in res
        assert len(res["per_query_results"]) == 1


def test_evaluate_shuffled_counterfactuals(synthetic_diagnostic_setup):
    dataset, video_scores, frame_accessor = synthetic_diagnostic_setup

    with patch("scripts.evaluation.run_counterfactuals.encode_query_events") as mock_encode:
        mock_encode.return_value = np.array(
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        )

        res = evaluate_shuffled_counterfactuals(
            dataset=dataset,
            video_scores=video_scores,
            frame_embeddings=frame_accessor,
            candidate_k=5,
            transition_weight=0.5,
            seed=42,
        )

        assert res["experiment"] == "shuffled_edge_counterfactual"
        assert res["num_queries"] == 1
        assert "shuffled_degradation_rate" in res
        assert "mean_margin" in res
        assert len(res["per_query_results"]) == 1


def test_save_counterfactual_result(tmp_path: Path):
    data = {"experiment": "test", "num_queries": 2, "metric": 0.85}
    out_file = tmp_path / "sub" / "result.json"

    save_counterfactual_result(data, out_file)

    assert out_file.is_file()
    with out_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data
