"""Tests for ablation experiment runner (SP-13)."""

from pathlib import Path
import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.diagnostic import DiagnosticDataset, DiagnosticEvent, DiagnosticQuery
from scripts.evaluation.run_motion_graph_ablation import (
    evaluate_method_on_dataset,
    save_summary_result,
)


def _make_sample_dataset_and_video() -> tuple[DiagnosticDataset, dict[str, VideoEventScores], dict[str, np.ndarray]]:
    ev1 = DiagnosticEvent(id="E1", text="sitting", target_frame_idxs=(0,), start_ms=1000, end_ms=2000)
    ev2 = DiagnosticEvent(id="E2", text="standing", target_frame_idxs=(2,), start_ms=3000, end_ms=4000)

    query = DiagnosticQuery(
        query_id="Q1",
        video_id="v1",
        query="sitting then standing",
        category="state_transition",
        events=(ev1, ev2),
    )
    dataset = DiagnosticDataset(queries=(query,))

    # Video with 3 frames
    scores = np.array([
        [0.9, 0.2, 0.1],  # ev1 peaks at f0 (idx 0)
        [0.1, 0.2, 0.9],  # ev2 peaks at f2 (idx 2)
    ])
    video = VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f0", "f1", "f2"]),
        frame_idx=np.array([0, 1, 2], dtype=np.int64),
        timestamps_ms=np.array([1000, 2000, 3000], dtype=np.int64),
        scores=scores,
    )
    video_scores = {"v1": video}

    # Frame embeddings (2D feature space)
    frame_embs = {
        "v1": np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ], dtype=np.float32)
    }

    return dataset, video_scores, frame_embs


@pytest.mark.parametrize("method_name", ["baseline_dp", "topk_no_edges", "visual_continuity", "ours"])
def test_evaluate_method_on_dataset(method_name: str, monkeypatch):
    dataset, video_scores, frame_embs = _make_sample_dataset_and_video()

    # Mock query encoding to avoid remote SigLIP download
    monkeypatch.setattr(
        "scripts.evaluation.run_motion_graph_ablation.encode_query_events",
        lambda texts, *args, **kwargs: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    summary = evaluate_method_on_dataset(
        method_name=method_name,
        dataset=dataset,
        video_scores=video_scores,
        frame_embeddings=frame_embs,
        candidate_k=3,
        transition_weight=0.25,
    )

    assert summary.method_name == method_name
    assert summary.num_queries == 1
    assert summary.r1 == 1.0
    assert summary.all_hit == 1.0
    assert summary.event_hit == 1.0
    assert summary.avg_latency_ms >= 0.0

    # Test table row formatting
    row = summary.to_table_row()
    assert method_name in row


def test_save_summary_result(tmp_path: Path, monkeypatch):
    dataset, video_scores, frame_embs = _make_sample_dataset_and_video()
    monkeypatch.setattr(
        "scripts.evaluation.run_motion_graph_ablation.encode_query_events",
        lambda texts, *args, **kwargs: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    summary = evaluate_method_on_dataset(
        method_name="baseline_dp",
        dataset=dataset,
        video_scores=video_scores,
        frame_embeddings=frame_embs,
    )

    out_file = tmp_path / "baseline_dp.json"
    save_summary_result(summary, out_file)
    assert out_file.is_file()
