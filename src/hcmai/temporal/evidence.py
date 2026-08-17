"""Canonical evidence adaptation and three-state evaluation bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hcmai.common.schemas import FrameEvidence, FrameRecord, RetrievalCandidate
from hcmai.data.pipeline import DataService

EvidenceKey = tuple[str, str]


class EvaluationState(str, Enum):
    """Derived evaluation state for one query-unit and candidate-video pair."""

    UNKNOWN = "unknown"
    EVALUATED_NO_MATCH = "evaluated_no_match"
    MATCHED = "matched"


@dataclass
class ProgressiveEvidenceState:
    """Derive UNKNOWN/empty/matched without storing a parallel status value."""

    evaluated_keys: set[EvidenceKey] = field(default_factory=set)
    evidence: dict[EvidenceKey, tuple[FrameEvidence, ...]] = field(default_factory=dict)

    def is_evaluated(self, unit_id: str, video_id: str) -> bool:
        """Return whether retrieval evaluated this unit within this video."""

        return (unit_id, video_id) in self.evaluated_keys

    def get_evidence(
        self,
        unit_id: str,
        video_id: str,
    ) -> tuple[FrameEvidence, ...]:
        """Return retained evidence without converting UNKNOWN into a score."""

        return self.evidence.get((unit_id, video_id), ())

    def evaluation_state(self, unit_id: str, video_id: str) -> EvaluationState:
        """Derive UNKNOWN, evaluated-empty, or matched from the two containers."""

        key = (unit_id, video_id)
        if key not in self.evaluated_keys:
            return EvaluationState.UNKNOWN
        if self.evidence.get(key):
            return EvaluationState.MATCHED
        return EvaluationState.EVALUATED_NO_MATCH

    def mark_evaluated(
        self,
        unit_id: str,
        video_id: str,
        items: tuple[FrameEvidence, ...] = (),
    ) -> None:
        """Record a completed evaluation and retain canonical matched evidence."""

        key = (unit_id, video_id)
        self.evaluated_keys.add(key)
        if items:
            self.evidence[key] = deduplicate_evidence(items)
        else:
            self.evidence.pop(key, None)
        self.validate()

    def unknown_units(self, unit_ids: list[str], video_id: str) -> list[str]:
        """Return only units that still require lazy backfill for a video."""

        return [
            unit_id
            for unit_id in unit_ids
            if not self.is_evaluated(unit_id, video_id)
        ]

    def retain_videos(self, video_ids: set[str]) -> None:
        """Bound state to the explicit active candidate pool."""

        self.evaluated_keys = {
            key for key in self.evaluated_keys if key[1] in video_ids
        }
        self.evidence = {
            key: items
            for key, items in self.evidence.items()
            if key[1] in video_ids
        }
        self.validate()

    def validate(self) -> None:
        """Enforce that every matched evidence entry is explicitly evaluated."""

        orphaned = set(self.evidence) - self.evaluated_keys
        if orphaned:
            raise ValueError(
                "matched evidence must be evaluated: "
                f"{sorted(orphaned)!r}"
            )
        if any(not items for items in self.evidence.values()):
            raise ValueError(
                "empty evaluated evidence must be represented by no dict entry"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluated-empty pairs separately from matched evidence."""

        self.validate()
        return {
            "evaluated_keys": [list(key) for key in sorted(self.evaluated_keys)],
            "evidence": [
                {
                    "unit_id": unit_id,
                    "video_id": video_id,
                    "items": [item.model_dump(mode="json") for item in items],
                }
                for (unit_id, video_id), items in sorted(self.evidence.items())
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProgressiveEvidenceState:
        """Restore evidence state while rechecking its hard invariants."""

        state = cls(
            evaluated_keys={tuple(item) for item in value.get("evaluated_keys", [])},
            evidence={
                (row["unit_id"], row["video_id"]): tuple(
                    FrameEvidence.model_validate(item) for item in row["items"]
                )
                for row in value.get("evidence", [])
            },
        )
        state.validate()
        return state


def retrieval_to_evidence(
    candidate: RetrievalCandidate,
    unit_id: str,
    data: DataService,
) -> FrameEvidence:
    """Resolve retrieval identity through the authoritative frame store."""

    frame = data.get_frame(candidate.frame_id)
    if not isinstance(frame, FrameRecord):
        frame = FrameRecord(
            frame_id=getattr(frame, "frame_id", candidate.frame_id),
            video_id=getattr(frame, "video_id", "unknown"),
            frame_idx=getattr(frame, "frame_idx", 0),
            timestamp_ms=getattr(frame, "timestamp_ms", 0),
            image_path=getattr(frame, "image_path", f"{candidate.frame_id}.jpg"),
            thumbnail_path=getattr(frame, "thumbnail_path", None),
            width=getattr(frame, "width", 640),
            height=getattr(frame, "height", 360),
            fps=getattr(frame, "fps", None),
        )
    if frame.frame_id != candidate.frame_id:
        raise ValueError("retrieval frame_id conflicts with canonical FrameRecord")
    _validate_duplicate_metadata(
        candidate,
        frame.frame_id,
        frame.video_id,
        frame.frame_idx,
        frame.timestamp_ms,
    )
    score = candidate.final_score
    if score is None:
        score = candidate.reranker_score
    if score is None:
        score = candidate.fusion_score
    if score is None:
        score = max(candidate.source_scores.values(), default=0.0)
    return FrameEvidence(
        frame=frame,
        unit_scores={unit_id: float(score)},
        source_scores=dict(candidate.source_scores),
        source_ranks=dict(candidate.source_ranks),
        score=float(score),
        provenance=tuple(source.value for source in candidate.source_scores),
    )


def deduplicate_evidence(items: tuple[FrameEvidence, ...]) -> tuple[FrameEvidence, ...]:
    """Deduplicate only by canonical frame_id and retain the strongest evidence."""

    by_id: dict[str, FrameEvidence] = {}
    for item in items:
        prior = by_id.get(item.frame.frame_id)
        if prior is None or item.score > prior.score:
            by_id[item.frame.frame_id] = item
    return tuple(sorted(
        by_id.values(),
        key=lambda item: (-item.score, item.frame.timestamp_ms, item.frame.frame_id),
    ))


def _validate_duplicate_metadata(
    candidate: RetrievalCandidate,
    frame_id: str,
    video_id: str,
    frame_idx: int,
    timestamp_ms: int,
) -> None:
    """Reject duplicated retrieval metadata that conflicts with canonical data."""

    expected = {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
    }
    nested = candidate.metadata.get("frame")
    frame_metadata = nested if isinstance(nested, dict) else {}
    for key, canonical in expected.items():
        supplied = candidate.metadata.get(key, frame_metadata.get(key))
        if supplied is not None and supplied != canonical:
            raise ValueError(
                f"retrieval metadata {key}={supplied!r} conflicts with "
                f"canonical {canonical!r}"
            )
