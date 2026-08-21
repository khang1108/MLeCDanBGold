"""Canonical deterministic Top-k VQA submission materialization."""

from __future__ import annotations

from hcmai.common.schemas import RetrievalSource, VQASubmission
from hcmai.common.utils.video import derive_fps, format_video_id, official_frame_idx
from ..domain.models import GroundedAnswerCandidate
from ..domain.ports import SubmissionData


def materialize_submissions(
    ranked: list[GroundedAnswerCandidate], data: SubmissionData, *, top_k: int = 100
) -> list[VQASubmission]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    rows: list[VQASubmission] = []
    seen: set[tuple[str, int, str]] = set()
    for candidate in ranked:
        try:
            frame = data.get_frame(candidate.evidence_frame_id)
        except KeyError:
            continue
        if frame.video_id != candidate.scene.video_id:
            continue
        key = (frame.video_id, frame.frame_idx, candidate.normalized_answer)
        if key in seen:
            continue
        seen.add(key)
        scene_evidence = getattr(candidate.scene, "evidence", None)
        scene_frame_ids = (
            [item.frame.frame_id for item in scene_evidence]
            if scene_evidence
            else [frame.frame_id]
        )
        frame_ids = (
            scene_frame_ids
            if frame.frame_id in scene_frame_ids
            else [frame.frame_id, *scene_frame_ids]
        )
        fps = derive_fps(frame)
        # ``frame_id`` selects the exact internal answer frame. ``frame_idx``
        # is copied from the BTC map for the official submission row.
        frame_idx = official_frame_idx(frame)
        rows.append(VQASubmission(
            rank=len(rows) + 1,
            video_id=format_video_id(frame.video_id, fallback_path=frame.image_path),
            frame_id=frame.frame_id,
            frame_ids=frame_ids,
            frame_idx=frame_idx,
            fps=fps,
            answer=candidate.answer,
            normalized_answer=candidate.normalized_answer,
            retrieval_score=candidate.frame_score,
            grounding_score=candidate.localization_score,
            answer_score=candidate.answer_confidence,
            joint_score=candidate.joint_score,
            timestamp_ms=frame.timestamp_ms,
            temporal_window=(candidate.scene.start_ms, candidate.scene.end_ms),
            evidence_consistency_score=candidate.consistency_score,
            provenance={"score_components": candidate.score_components},
            warnings=list(candidate.warnings),
            caption=data.get_evidence(frame.frame_id, RetrievalSource.CAPTION),
            evidence_summary=f"window={candidate.scene.start_ms}-{candidate.scene.end_ms}ms",
        ))
        if len(rows) == top_k:
            break
    return rows
