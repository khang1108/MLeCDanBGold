from hcmai.common.schemas import VQASubmission
from hcmai.evaluation.vqa import VQAGold, evaluate_vqa


def _row(rank: int, *, frame_idx: int, answer: str) -> VQASubmission:
    return VQASubmission(
        rank=rank,
        video_id="video-1",
        frame_id=f"frame-{rank}",
        frame_idx=frame_idx,
        answer=answer,
        normalized_answer=answer.lower(),
        retrieval_score=1.0,
        grounding_score=1.0,
        answer_score=1.0,
        joint_score=1.0,
    )


def test_vqa_metrics_require_video_frame_and_answer_in_the_same_row() -> None:
    gold = VQAGold("video-1", 10, 20, frozenset({"red"}))
    metrics = evaluate_vqa(
        [_row(1, frame_idx=5, answer="red"), _row(2, frame_idx=15, answer="blue")],
        gold,
    )

    assert metrics.correct_video == 1
    assert metrics.frame_interval == 1
    assert metrics.normalized_answer == 1
    assert metrics.joint == 0
    assert metrics.mean_top_k_score == 0


def test_vqa_metrics_respect_ranked_cutoffs() -> None:
    rows = [_row(rank, frame_idx=1, answer="blue") for rank in range(1, 6)]
    rows.append(_row(6, frame_idx=12, answer="red"))

    metrics = evaluate_vqa(rows, VQAGold("video-1", 10, 20, frozenset({"red"})))

    assert metrics.top_k_joint == {1: 0, 5: 0, 20: 1, 50: 1, 100: 1}
    assert metrics.mean_top_k_score == 0.6
