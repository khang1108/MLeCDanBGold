"""Canonical deterministic Top-k VQA submission materialization."""

from __future__ import annotations

from hcmai.common.schemas import VQASubmission
from .contracts import FrameLookup
from .models import GroundedAnswerCandidate


def materialize_submissions(
    ranked: list[GroundedAnswerCandidate], data: FrameLookup, *, top_k: int = 100
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
        if frame.video_id != candidate.window.video_id:
            continue
        key = (frame.video_id, frame.frame_idx, candidate.normalized_answer)
        if key in seen:
            continue
        seen.add(key)
        rows.append(VQASubmission(
            rank=len(rows) + 1, video_id=frame.video_id, frame_id=frame.frame_id,
            frame_idx=frame.frame_idx, answer=candidate.answer,
            normalized_answer=candidate.normalized_answer,
            retrieval_score=candidate.frame_score,
            grounding_score=candidate.localization_score,
            answer_score=candidate.answer_confidence,
            joint_score=candidate.joint_score,
            timestamp_ms=frame.timestamp_ms,
            temporal_window=(candidate.window.start_ms, candidate.window.end_ms),
            evidence_consistency_score=candidate.consistency_score,
            provenance={"score_components": candidate.score_components},
            warnings=list(candidate.warnings),
            evidence_summary=f"window={candidate.window.start_ms}-{candidate.window.end_ms}ms",
        ))
        if len(rows) == top_k:
            break
    return rows
